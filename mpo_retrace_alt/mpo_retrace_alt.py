from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from mpo_retrace_alt.core import MLPActorCritic
from common.tray_dyn_buf import DynamicTrajectoryBuffer
from common.retrace import Retrace

local_device = "cpu"

def mpo_retrace(
        env,
        eps_dual=0.1,
        eps_mean=0.1,
        eps_cov=0.0001,
        gamma=0.99,
        alpha=10.,
        sample_episode_num=1,
        max_ep_len=200,
        len_rollout=200,
        sample_action_num=20,
        batch_size=768,
        batch_q=1,
        episode_rerun_num=20,
        epochs=20,
        lagrange_iteration_num=5,
        q_iteration_num=15,
        polyak=0.995):
    run = 0
    writer = SummaryWriter(comment='MPO_no_eta_reset')
    iteration = 0

    ac = MLPActorCritic(env)
    ac_targ = deepcopy(ac)

    actor = ac.pi
    critic = ac.q
    target_actor = ac_targ.pi
    target_critic = ac_targ.q

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-4)
    norm_loss_q = nn.SmoothL1Loss()

    eta = 1.0
    eta_mean = 0.0
    eta_cov = 0.0
    ds = env.observation_space.shape[0]
    da = env.action_space.shape[0]

    replay_buffer = DynamicTrajectoryBuffer(ds,
                                            da,
                                            max_ep_len,
                                            max_ep_len,
                                            len_rollout,
                                            5000,
                                            local_device)

    def gaussian_kl(targ_mean, mean, targ_std, std):
        n = std.size(-1)
        cov = std ** 2
        targ_cov = targ_std ** 2
        cov_inv = 1 / cov
        targ_cov_inv = 1 / targ_cov
        inner_mean = (((mean - targ_mean) ** 2) * targ_cov_inv).sum(-1)
        inner_cov = torch.log(cov.prod(-1) / targ_cov.prod(-1)) - n + (cov_inv * targ_cov).sum(-1)
        c_mean = 0.5 * torch.mean(inner_mean)
        c_cov = 0.5 * torch.mean(inner_cov)
        return c_mean, c_cov

    def update_q_retrace(samples, run):
        critic_optimizer.zero_grad()

        batch_q = critic.forward(samples['state'], samples['action'])
        batch_q = torch.transpose(batch_q, 0, 1)

        targ_q = target_critic.forward(samples['state'], samples['action'])
        targ_q = torch.transpose(targ_q, 0, 1)

        # technically wrong: after debugging actor should be changed to target_actor
        for p in actor.parameters():
            p.requires_grad = False

        targ_mean, targ_chol = actor.forward(samples['state'])
        targ_act = get_act(targ_mean, targ_chol)

        exp_targ_q = target_critic.forward(samples['state'], targ_act)
        exp_targ_q = torch.transpose(exp_targ_q, 0, 1)

        targ_act_logp = get_logp(targ_mean, targ_chol, samples['action']).unsqueeze(-1)
        targ_act_logp = torch.transpose(targ_act_logp, 0, 1)

        # technically wrong
        for p in actor.parameters():
            p.requires_grad = True

        retrace = Retrace()
        loss_q = retrace(Q=batch_q,
                         expected_target_Q=exp_targ_q,
                         target_Q=targ_q,
                         rewards=torch.transpose(samples['reward'], 0, 1),
                         target_policy_probs=targ_act_logp,
                         behaviour_policy_probs=torch.transpose(samples['pi_logp'], 0, 1),
                         gamma=gamma
                         )
        writer.add_scalar('q_loss', loss_q.item(), run)
        writer.add_scalar('q', targ_q.detach().numpy().mean(), run)
        writer.add_scalar('q_min', targ_q.detach().numpy().min(), run)
        writer.add_scalar('q_max', targ_q.detach().numpy().max(), run)

        loss_q.backward()
        critic_optimizer.step()

    def sample_traj(perform_traj):
        for _ in range(perform_traj):

            # sample steps
            s, ep_ret, ep_len = env.reset(), 0, 0
            while True:
                a, logp = get_action(s)
                # do step in environment
                s2, r, d, _ = env.step(a.reshape(1, 1).numpy())
                ep_ret += r
                ep_len += 1

                d = False if ep_len == max_ep_len else d

                replay_buffer.store(s.reshape(ds), s2.reshape(ds), a, r, logp.cpu().numpy(), d)
                s = s2
                # reset environment (ignore done signal)
                if ep_len == max_ep_len:
                    s, ep_ret, ep_len = env.reset(), 0, 0
                    replay_buffer.next_traj()
                    break

    def get_action(state, deterministic=False):
        mean, chol = target_actor.forward(torch.as_tensor(state,
                                                               dtype=torch.float32,
                                                               device=local_device).reshape(1, ds))
        if deterministic:
            return mean, None
        act = get_act(mean, chol).squeeze()
        return act, get_logp(mean, chol, act)

    def test_agent(run):
        ep_ret_list = list()
        for _ in tqdm(range(200), desc="testing model"):
            s, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                with torch.no_grad():
                    s, r, d, _ = env.step(
                        get_action(s, deterministic=True)[0].reshape(1, 1).numpy())
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
        writer.add_scalar('test_ep_ret', np.array(ep_ret_list).mean(), run)
        print('test_ep_ret:', np.array(ep_ret_list).mean(), ' ', run)

    def get_logp(mean, cov, action, expand=None):
        if expand is not None:
            dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=da).expand(expand)
        else:
            dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=da)
        return dist.log_prob(action)

    def get_act(mean, cov, amount=None):
        dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=da)
        if amount is not None:
            return dist.sample(amount)
        return dist.sample()

    for it in range(iteration, epochs):
        # Find better policy by gradient descent
        for r in tqdm(range(episode_rerun_num * 25), desc='updating nets'):
            if r % 25 == 0:
                if it == 0 and r == 0:
                    # update replay buffer
                    sample_traj(perform_traj=10)
                else:
                    sample_traj(perform_traj=sample_episode_num)

            # update q values
            if r % 50 == 0:
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(param.data)

            B = batch_size
            M = sample_action_num

            # update q with retrace
            for _ in range(q_iteration_num):
                rows, cols = replay_buffer.sample_idxs(batch_size=batch_q)
                samples = replay_buffer.sample_trajectories(rows, cols)
                update_q_retrace(samples, run)

            rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_size)
            samples = replay_buffer.sample_batch(rows, cols)
            state_batch = samples['state']

            # sample M additional action for each state
            with torch.no_grad():
                b_μ, b_A = target_actor.forward(state_batch)  # (B,)
                sampled_actions = get_act(b_μ, b_A, amount=(M,))
                expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
                target_q = target_critic.forward(
                    expanded_states.reshape(-1, ds),  # (M * B, ds)
                    sampled_actions.reshape(-1, da)  # (M * B, da)
                ).reshape(M, B)  # (M, B)
                target_q_np = target_q.cpu().numpy()  # (M, B)

            # E-step
            def dual(eta):
                max_q = np.max(target_q_np, 0)
                return eta * eps_dual + np.mean(max_q) \
                       + eta * np.mean(np.log(np.mean(np.exp((target_q_np - max_q) / eta), axis=0)))

            bounds = [(1e-6, None)]
            res = minimize(dual, np.array([eta]), method='SLSQP', bounds=bounds)
            eta = res.x[0]
            writer.add_scalar('eta', eta, run)

            qij = torch.softmax(target_q / eta, dim=0)  # (M, B) or (da, B)

            # M-step
            # update policy based on lagrangian
            for _ in range(lagrange_iteration_num):
                mean, std = actor.forward(state_batch)
                loss_p = torch.mean(
                    qij * (
                            get_logp(mean, b_A, sampled_actions, expand=(M, B)) +
                            get_logp(b_μ, std, sampled_actions, expand=(M, B))
                    )
                )
                writer.add_scalar('loss_pi', loss_p.item(), run)

                c_mean, c_cov = gaussian_kl(
                    targ_mean=b_μ, mean=mean,
                    targ_std=b_A, std=std)

                # Update lagrange multipliers by gradient descent
                eta_mean -= alpha * (eps_mean - c_mean).detach().item()
                if eta_mean < 0.0:
                    eta_mean = 0.0
                writer.add_scalar('eta_mean', eta_mean, run)

                eta_cov -= alpha * (eps_cov - c_cov).detach().item()
                if eta_cov < 0.0:
                    eta_cov = 0.0
                writer.add_scalar('eta_cov', eta_cov, run)

                actor_optimizer.zero_grad()
                loss_l = -(
                        loss_p
                        + eta_mean * (eps_mean - c_mean)
                        + eta_cov * (eps_cov - c_cov)
                )
                writer.add_scalar('combined_loss', loss_l.item(), run)
                loss_l.backward()
                clip_grad_norm_(actor.parameters(), 0.1)
                actor_optimizer.step()

                if r % 50 == 0 and r >= 1:
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(param.data)
                run += 1

        test_agent(it)
        writer.flush()
        it += 1
