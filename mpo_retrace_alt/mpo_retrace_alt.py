from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
# from mpo_retrace_alt.actor import ActorContinuous
# from mpo_retrace_alt.critic import CriticContinuous
from mpo_retrace_alt.core import MLPActorCritic
from common.tray_dyn_buf import DynamicTrajectoryBuffer
from common.retrace import Retrace

local_device = "cpu"


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(targ_mean, mean, targ_std, std):
    n = std.size(-1)
    cov = std ** 2
    targ_cov = targ_std ** 2
    cov_inv = 1 / cov
    targ_cov_inv = 1 / targ_cov
    inner_mean = (((mean - targ_mean) ** 2) * targ_cov_inv).sum(-1)
    inner_cov = torch.log(cov.prod(-1) / targ_cov.prod(-1)) - n + (cov_inv * targ_cov).sum(-1)
    C_μ = 0.5 * torch.mean(inner_mean)
    C_Σ = 0.5 * torch.mean(inner_cov)
    return C_μ, C_Σ


class MPO(object):
    def __init__(self,
                 device,
                 env,
                 dual_constraint=0.1,
                 kl_mean_constraint=0.1,
                 kl_var_constraint=0.0001,
                 discount_factor=0.99,
                 alpha=10,
                 sample_episode_num=1,
                 max_ep_len=200,
                 len_rollout=200,
                 sample_action_num=20,
                 batch_size=768,
                 batch_q=1,
                 episode_rerun_num=20,
                 lagrange_iteration_num=5,
                 update_q=15,
                 polyak=0.995,
                 ):
        self.writer = SummaryWriter(comment='MPO_RETRACE_ALT_FREQ_UPDATES_LESS_ITERATIONS')
        self.device = device
        self.env = env
        self.continuous_action_space = True
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]

        self.eps_dual = dual_constraint
        self.eps_mean = kl_mean_constraint  # hard constraint for the KL
        self.eps_cov = kl_var_constraint  # hard constraint for the KL
        self.gamma = discount_factor
        self.alpha = alpha  # scaling factor for the update step of η_μ
        self.sample_episode_num = sample_episode_num
        self.sample_action_num = sample_action_num
        self.batch_size = batch_size
        self.episode_rerun_num = episode_rerun_num
        self.lagrange_iteration_num = lagrange_iteration_num
        self.batch_q = batch_q
        self.update_q = update_q
        self.max_ep_len = max_ep_len
        self.polyak = polyak

        ac = MLPActorCritic(env)
        ac_targ = deepcopy(ac)

        self.actor = ac.pi
        self.critic = ac.q
        self.target_actor = ac_targ.pi
        self.target_critic = ac_targ.q

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2e-4)
        self.norm_loss_q = nn.SmoothL1Loss()

        self.eta = 1.0
        self.eta_mean = 0.0
        self.eta_cov = 0.0

        self.replay_buffer = DynamicTrajectoryBuffer(self.ds,
                                                     self.da,
                                                     max_ep_len,
                                                     max_ep_len,
                                                     len_rollout,
                                                     5000,
                                                     local_device)

        self.iteration = 0
        self.render = False

    def train(self,
              iteration_num=20,
              ):
        writer = self.writer
        run = 0

        for it in range(self.iteration, iteration_num):
            # Find better policy by gradient descent
            for r in tqdm(range(self.episode_rerun_num * 25), desc='updating nets'):
                if r % 25 == 0:
                    if it == 0 and r == 0:
                        # update replay buffer
                        self.sample_traj(perform_traj=10)
                    else:
                        self.sample_traj(perform_traj=1)

                # update q values
                if r % 50 == 0:
                    for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                        target_param.data.copy_(param.data)

                B = self.batch_size
                M = self.sample_action_num
                ds = self.ds
                da = self.da

                # update q with retrace
                for _ in range(self.update_q):
                    rows, cols = self.replay_buffer.sample_idxs(batch_size=self.batch_q)
                    samples = self.replay_buffer.sample_trajectories(rows, cols)
                    self.update_q_retrace(samples, run)

                rows, cols = self.replay_buffer.sample_idxs_batch(batch_size=self.batch_size)
                samples = self.replay_buffer.sample_batch(rows, cols)
                state_batch = samples['state']

                # sample M additional action for each state
                with torch.no_grad():
                    b_μ, b_A = self.target_actor.forward(state_batch)  # (B,)
                    sampled_actions = self.get_act(b_μ, b_A, amount=(M,))
                    expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
                    target_q = self.target_critic.forward(
                        expanded_states.reshape(-1, ds),  # (M * B, ds)
                        sampled_actions.reshape(-1, da)  # (M * B, da)
                    ).reshape(M, B)  # (M, B)
                    target_q_np = target_q.cpu().numpy()  # (M, B)

                # E-step
                def dual(eta):
                    max_q = np.max(target_q_np, 0)
                    return eta * self.eps_dual + np.mean(max_q) \
                           + eta * np.mean(np.log(np.mean(np.exp((target_q_np - max_q) / eta), axis=0)))

                bounds = [(1e-6, None)]
                res = minimize(dual, np.array([self.eta]), method='SLSQP', bounds=bounds)
                self.eta = res.x[0]
                writer.add_scalar('eta', self.eta, run)

                qij = torch.softmax(target_q / self.eta, dim=0)  # (M, B) or (da, B)

                # M-step
                # update policy based on lagrangian
                for _ in range(self.lagrange_iteration_num):
                    mean, std = self.actor.forward(state_batch)
                    loss_p = torch.mean(
                        qij * (
                                self.get_logp(mean, b_A, sampled_actions, expand=(M, B)) +
                                self.get_logp(b_μ, std, sampled_actions, expand=(M, B))
                        )
                    )
                    writer.add_scalar('loss_pi', loss_p.item(), run)

                    kl_mean, kl_cov = gaussian_kl(
                        targ_mean=b_μ, mean=mean,
                        targ_std=b_A, std=std)

                    # Update lagrange multipliers by gradient descent
                    self.eta_mean -= self.alpha * (self.eps_mean - kl_mean).detach().item()
                    if self.eta_mean < 0.0:
                        self.eta_mean = 0.0
                    writer.add_scalar('eta_mean', self.eta_mean, run)

                    self.eta_cov -= self.alpha * (self.eps_cov - kl_cov).detach().item()
                    if self.eta_cov < 0.0:
                        self.eta_cov = 0.0
                    writer.add_scalar('eta_cov', self.eta_cov, run)

                    self.actor_optimizer.zero_grad()
                    loss_l = -(
                            loss_p
                            + self.eta_mean * (self.eps_mean - kl_mean)
                            + self.eta_cov * (self.eps_cov - kl_cov)
                    )
                    writer.add_scalar('combined_loss', loss_l.item(), run)
                    loss_l.backward()
                    clip_grad_norm_(self.actor.parameters(), 0.1)
                    self.actor_optimizer.step()

                    if r % 50 == 0 and r >= 1:
                        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                            target_param.data.copy_(param.data)
                    run += 1

            self.eta_mean = 0.0
            self.eta_cov = 0.0

            self.test_agent(it)
            writer.flush()
            it += 1

    def update_q_retrace(self, samples, run):
        self.critic_optimizer.zero_grad()

        batch_q = self.critic.forward(samples['state'], samples['action'])
        batch_q = torch.transpose(batch_q, 0, 1)

        targ_q = self.target_critic.forward(samples['state'], samples['action'])
        targ_q = torch.transpose(targ_q, 0, 1)

        # technically wrong: after debugging actor should be changed to target_actor
        for p in self.actor.parameters():
            p.requires_grad = False

        targ_mean, targ_chol = self.actor.forward(samples['state'])
        targ_act = self.get_act(targ_mean, targ_chol)

        exp_targ_q = self.target_critic.forward(samples['state'], targ_act)
        exp_targ_q = torch.transpose(exp_targ_q, 0, 1)

        targ_act_logp = self.get_logp(targ_mean, targ_chol, samples['action']).unsqueeze(-1)
        targ_act_logp = torch.transpose(targ_act_logp, 0, 1)

        # technically wrong
        for p in self.actor.parameters():
            p.requires_grad = True

        retrace = Retrace()
        loss_q = retrace(Q=batch_q,
                         expected_target_Q=exp_targ_q,
                         target_Q=targ_q,
                         rewards=torch.transpose(samples['reward'], 0, 1),
                         target_policy_probs=targ_act_logp,
                         behaviour_policy_probs=torch.transpose(samples['pi_logp'], 0, 1),
                         gamma=self.gamma
                         )
        self.writer.add_scalar('q_loss', loss_q.item(), run)
        self.writer.add_scalar('q', targ_q.detach().numpy().mean(), run)
        self.writer.add_scalar('q_min', targ_q.detach().numpy().min(), run)
        self.writer.add_scalar('q_max', targ_q.detach().numpy().max(), run)

        loss_q.backward()
        self.critic_optimizer.step()

    def sample_traj(self, perform_traj):
        for _ in range(perform_traj):

            # sample steps
            s, ep_ret, ep_len = self.env.reset(), 0, 0
            while True:
                a, logp = self.get_action(s)
                # do step in environment
                s2, r, d, _ = self.env.step(a.reshape(1, 1).numpy())
                ep_ret += r
                ep_len += 1

                d = False if ep_len == self.max_ep_len else d

                self.replay_buffer.store(s.reshape(self.ds), s2.reshape(self.ds), a, r, logp.cpu().numpy(), d)
                s = s2
                # reset environment (ignore done signal)
                if ep_len == self.max_ep_len:
                    s, ep_ret, ep_len = self.env.reset(), 0, 0
                    self.replay_buffer.next_traj()
                    break

    def get_action(self, state, deterministic=False):
        mean, chol = self.target_actor.forward(torch.as_tensor(state,
                                                               dtype=torch.float32,
                                                               device=local_device).reshape(1, self.ds))
        if deterministic:
            return mean, None
        act = self.get_act(mean, chol).squeeze()
        return act, self.get_logp(mean, chol, act)

    def test_agent(self, run):
        ep_ret_list = list()
        for _ in tqdm(range(200), desc="testing model"):
            s, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                with torch.no_grad():
                    s, r, d, _ = self.env.step(
                        self.get_action(s, deterministic=True)[0].reshape(1, 1).numpy())
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
        self.writer.add_scalar('test_ep_ret', np.array(ep_ret_list).mean(), run)
        print('test_ep_ret:', np.array(ep_ret_list).mean(), ' ', run)

    def get_logp(self, mean, cov, action, expand=None):
        if expand is not None:
            dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=self.da).expand(expand)
        else:
            dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=self.da)
        return dist.log_prob(action)

    def get_act(self, mean, cov, amount=None):
        dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=self.da)
        if amount is not None:
            return dist.sample(amount)
        return dist.sample()
