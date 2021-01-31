import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal
from mpo_retrace_alt.actor import ActorContinuous
from mpo_retrace_alt.critic import CriticContinuous
from common.tray_dyn_buf import DynamicTrajectoryBuffer
from common.retrace import Retrace
from utils.logx import EpochLogger

local_device = "cpu"


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(μi, μ, Ai, A):
    n = A.size(-1)
    μi = μi.unsqueeze(-1)  # (B, n, 1)
    μ = μ.unsqueeze(-1)  # (B, n, 1)
    Σi = Ai @ bt(Ai)  # (B, n, n)
    Σ = A @ bt(A)  # (B, n, n)
    Σi_inv = Σi.inverse()  # (B, n, n)
    Σ_inv = Σ.inverse()  # (B, n, n)
    inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
    inner_Σ = torch.log(Σ.det() / Σi.det()) - n + btr(Σ_inv @ Σi)  # (B,)
    C_μ = 0.5 * torch.mean(inner_μ)
    C_Σ = 0.5 * torch.mean(inner_Σ)
    return C_μ, C_Σ


def mpo_retrace(writer,
                env,
                eps_dual=0.1,
                eps_mean=0.1,
                eps_cov=0.0001,
                gamma=0.99,
                alpha=10.,
                lr_q=2e-4,
                lr_pi=3e-4,
                sample_episodes=1,
                episode_len=200,
                batch_act=20,
                batch_s=768,
                batch_t=1,
                episode_rerun_num=20,
                epochs=20,
                q_iteration_num=10,
                lagrange_iteration_num=5,
                update_q_after=100,
                update_pi_after=100,
                ):
    logger = EpochLogger()
    ds = env.observation_space.shape[0]
    da = env.action_space.shape[0]

    actor = ActorContinuous(env).to(local_device)
    critic = CriticContinuous(env).to(local_device)
    target_actor = ActorContinuous(env).to(local_device)
    target_critic = CriticContinuous(env).to(local_device)

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)
        target_param.requires_grad = False

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_pi)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_q)

    eta = 1.0
    eta_mean = 0.0
    eta_cov = 0.0

    replay_buffer = DynamicTrajectoryBuffer(ds,
                                            da,
                                            episode_len,
                                            episode_len,
                                            episode_len,
                                            10000,
                                            local_device)

    iteration = 0
    run = 0
    # logger.setup_pytorch_saver()

    def update_q_retrace(samples, run):
        critic_optimizer.zero_grad()

        batch_q = critic.forward(samples['state'], samples['action'], dim=2)
        batch_q = torch.transpose(batch_q, 0, 1)

        targ_q = target_critic.forward(samples['state'], samples['action'], dim=2)
        targ_q = torch.transpose(targ_q, 0, 1)

        targ_mean, targ_chol = actor.forward(samples['state'], traj=True, T=batch_t)
        dist = MultivariateNormal(targ_mean, scale_tril=targ_chol)
        targ_act = dist.sample()

        exp_targ_q = target_critic.forward(samples['state'], targ_act, dim=2)
        exp_targ_q = torch.transpose(exp_targ_q, 0, 1)

        targ_act_logp = dist.log_prob(samples['action']).unsqueeze(-1)
        targ_act_logp = torch.transpose(targ_act_logp, 0, 1)

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

                d = False if ep_len == episode_len else d

                replay_buffer.store(
                    s.reshape(ds), s2.reshape(ds), a, r, logp.cpu().numpy(), d)
                s = s2
                # reset environment (ignore done signal)
                if ep_len == episode_len:
                    s, ep_ret, ep_len = env.reset(), 0, 0
                    replay_buffer.next_traj()
                    break

    def get_action(state, deterministic=False):
        mean, chol = target_actor.forward(torch.as_tensor(state,
                                                          dtype=torch.float32,
                                                          device=local_device).reshape(1, ds))
        if deterministic:
            return mean, None
        dist = MultivariateNormal(mean, scale_tril=chol)
        act = dist.sample().squeeze()
        return act, dist.log_prob(act)

    def test_agent(run):
        ep_ret_list = list()
        for _ in tqdm(range(200), desc="testing model"):
            s, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not (d or (ep_len == episode_len)):
                with torch.no_grad():
                    s, r, d, _ = env.step(
                        get_action(s, deterministic=True)[0].reshape(1, 1).numpy())
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
        writer.add_scalar('test_ep_ret', np.array(ep_ret_list).mean(), run)
        print('test_ep_ret:', np.array(ep_ret_list).mean(), ' ', run)

    # main loop
    for it in range(iteration, epochs):
        # Find better policy by gradient descent
        for r in tqdm(range(episode_rerun_num * 50), desc='updating nets'):
            if r % 50 == 0:
                if it == 0 and r == 0:
                    # update replay buffer
                    sample_traj(perform_traj=sample_episodes)

            # update q values
            if r % update_q_after == 0:
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(param.data)

            B = batch_s
            M = batch_act

            # update q with retrace
            for _ in range(q_iteration_num):
                rows, cols = replay_buffer.sample_idxs(batch_size=batch_t)
                samples = replay_buffer.sample_trajectories(rows, cols)
                update_q_retrace(samples, run)

            rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_s)
            samples = replay_buffer.sample_batch(rows, cols)
            state_batch = samples['state']

            # sample M additional action for each state
            with torch.no_grad():
                b_μ, b_A = target_actor.forward(state_batch)  # (B,)
                b = MultivariateNormal(b_μ, scale_tril=b_A)  # (B,)
                sampled_actions = b.sample((M,))  # (M, B, da)
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
                μ, A = actor.forward(state_batch)
                π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (B,)
                π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (B,)
                loss_p = torch.mean(
                    qij * (
                            π1.expand((M, B)).log_prob(sampled_actions)  # (M, B)
                            + π2.expand((M, B)).log_prob(sampled_actions)  # (M, B)
                    )
                )
                writer.add_scalar('loss_pi', loss_p.item(), run)

                kl_mean, kl_cov = gaussian_kl(
                    μi=b_μ, μ=μ,
                    Ai=b_A, A=A)

                # Update lagrange multipliers by gradient descent
                eta_mean -= alpha * (eps_mean - kl_mean).detach().item()
                if eta_mean < 0.0:
                    eta_mean = 0.0
                writer.add_scalar('eta_mean', eta_mean, run)

                eta_cov -= alpha * (eps_cov - kl_cov).detach().item()
                if eta_cov < 0.0:
                    eta_cov = 0.0
                writer.add_scalar('eta_cov', eta_cov, run)

                actor_optimizer.zero_grad()
                loss_l = -(
                        loss_p
                        + eta_mean * (eps_mean - kl_mean)
                        + eta_cov * (eps_cov - kl_cov)
                )
                writer.add_scalar('combined_loss', loss_l.item(), run)
                loss_l.backward()
                clip_grad_norm_(actor.parameters(), 0.1)
                actor_optimizer.step()

                if r % update_pi_after == 0 and r >= 1:
                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(param.data)
                run += 1

        test_agent(it)
        writer.flush()
        it += 1
