from copy import deepcopy
from itertools import chain
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Independent, Normal
from mpo_retrace.core import MLPActorCritic
from common.tray_dyn_buf import DynamicTrajectoryBuffer
from common.retrace import Retrace


def mpo_retrace(
        env,
        writer,
        eps_dual=0.1,
        eps_mean=0.1,
        eps_cov=0.0001,
        gamma=0.99,
        alpha=10.,
        lr_pi=5e-4,
        lr_q=2e-4,
        sample_episodes=1,
        episode_len=200,
        batch_act=20,
        batch_s=768,
        batch_q=1,
        episodes=20,
        epochs=20,
        update_times_pi=5,
        update_times_q=15,
        update_q_after=50,
        update_pi_after=25,
        local_device='cuda:0'):
    run = 0
    iteration = 0

    eta = torch.tensor([1.0], device=local_device, requires_grad=True)
    eta_mean = 0.0
    eta_cov = 0.0

    ac = MLPActorCritic(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False

    actor = ac.pi
    critic = ac.q
    target_actor = ac_targ.pi
    target_critic = ac_targ.q

    actor_optimizer = torch.optim.Adam(chain(actor.parameters(), [eta]), lr=lr_pi)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_q)

    ds = env.observation_space.shape[0]
    da = env.action_space.shape[0]

    replay_buffer = DynamicTrajectoryBuffer(ds,
                                            da,
                                            episode_len,
                                            episode_len,
                                            episode_len,
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

        targ_mean, targ_chol = target_actor.forward(samples['state'])
        targ_act = get_act(targ_mean, targ_chol)

        exp_targ_q = target_critic.forward(samples['state'], targ_act)
        exp_targ_q = torch.transpose(exp_targ_q, 0, 1)

        targ_act_logp = get_logp(targ_mean, targ_chol, samples['action']).unsqueeze(-1)
        targ_act_logp = torch.transpose(targ_act_logp, 0, 1)

        retrace = Retrace(device=local_device)
        loss_q = retrace(Q=batch_q,
                         expected_target_Q=exp_targ_q,
                         target_Q=targ_q,
                         rewards=torch.transpose(samples['reward'], 0, 1),
                         target_policy_probs=targ_act_logp,
                         behaviour_policy_probs=torch.transpose(samples['pi_logp'], 0, 1),
                         gamma=gamma
                         )
        writer.add_scalar('q_loss', loss_q.item(), run)
        writer.add_scalar('q', targ_q.detach().mean().item(), run)
        writer.add_scalar('q_min', targ_q.detach().min().item(), run)
        writer.add_scalar('q_max', targ_q.detach().max().item(), run)

        loss_q.backward()
        critic_optimizer.step()

    def sample_traj(perform_traj):
        for i in range(perform_traj):

            # sample steps
            s, _, ep_len = env.reset(), 0, 0
            while True:
                a, logp = get_action(s)
                # do step in environment
                s2, r, d, _ = env.step(a.reshape(1, da).cpu().numpy())
                ep_len += 1

                replay_buffer.store(s.reshape(ds), s2.reshape(ds), a.cpu().numpy(), r, logp.cpu().numpy(), d)
                s = s2
                # reset environment (ignore done signal)
                if ep_len == episode_len:
                    s, _, ep_len = env.reset(), 0, 0
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
            while not (d or (ep_len == episode_len)):
                with torch.no_grad():
                    s, r, d, _ = env.step(
                        get_action(s, deterministic=True)[0].reshape(1, da).cpu().numpy())
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
        writer.add_scalar('test_ep_ret', np.array(ep_ret_list).mean(), run)
        print('test_ep_ret:', np.array(ep_ret_list).mean(), ' ', run)

    def get_logp(mean, cov, action, expand=None):
        if expand is not None:
            dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=1).expand(expand)
        else:
            dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=1)
        return dist.log_prob(action)

    def get_act(mean, cov, amount=None):
        dist = Independent(Normal(mean, cov), reinterpreted_batch_ndims=1)
        if amount is not None:
            return dist.sample(amount)
        return dist.sample()

    def dual(eta):
        max_q = torch.max(target_q, dim=0).values
        return eta * eps_dual + torch.mean(max_q) \
               + eta * torch.mean(torch.log(torch.mean(torch.exp((target_q - max_q) / eta), dim=0)))

    for it in range(iteration, epochs):
        # Find better policy by gradient descent
        for r in tqdm(range(episodes * 50), desc='updating nets'):
            if r % 50 == 0:
                # update replay buffer
                if it == 0 and r == 0:
                    # for better start sample more trajectories
                    sample_traj(perform_traj=10)
                else:
                    sample_traj(perform_traj=sample_episodes)

            # update q values
            if r % update_q_after == 0:
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(param.data)

            B = batch_s
            M = batch_act

            # update q with retrace
            for _ in range(update_times_q):
                rows, cols = replay_buffer.sample_idxs(batch_size=batch_q)
                samples = replay_buffer.sample_trajectories(rows, cols)
                update_q_retrace(samples, run)

            rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_s)
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

            # M-step
            qij = torch.softmax(target_q / eta.item(), dim=0)  # (M, B) or (da, B)
            # update policy based on lagrangian
            for _ in range(update_times_pi):
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

                # learn eta together with other policy parameters

                actor_optimizer.zero_grad()
                loss_eta = dual(eta)
                loss_l = -(
                        loss_p
                        + eta_mean * (eps_mean - c_mean)
                        + eta_cov * (eps_cov - c_cov)
                ) + loss_eta

                writer.add_scalar('combined_loss', loss_l.item() - loss_eta.item(), run)
                writer.add_scalar('eta', eta.item(), run)
                writer.add_scalar('eta_loss', loss_eta.item(), run)

                loss_l.backward()
                clip_grad_norm_(actor.parameters(), 0.1)
                actor_optimizer.step()

                if r % update_pi_after == 0 and r >= 1:
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(param.data)
                run += 1

        test_agent(it)
        writer.flush()
        it += 1
