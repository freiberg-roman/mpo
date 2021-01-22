from copy import deepcopy
import numpy as np
import itertools
import torch
from torch.optim import Adam
import time
from mpo_rerpi_style import core
from torch.utils.tensorboard import SummaryWriter
from mpo_rerpi_style.tray_dyn_buf import DynamicTrajectoryBuffer
from tqdm import tqdm
from mpo_algorithm.retrace import Retrace
from torch.distributions.normal import Normal

local_device = "cpu"


def mpo_td(env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        gamma=0.99,
        epochs=2000,
        traj_update_count=20,
        max_ep_len=200,
        eps=0.1,
        eps_mean=0.1,
        eps_cov=0.0001,
        lr=0.0003,
        alpha=1.,
        batch_q=3072,
        batch_s=3072,
        batch_t=32,
        batch_act=20,
        len_rollout=200,
        init_eta=0.5,
        init_eta_mean=1.0,
        init_eta_cov=1.0,
        runs_update_q=20,
        runs_update_pi=20,
        num_test_episodes=10,
        reward_scaling=lambda r: r,
        polyak=0.995):
    writer = SummaryWriter(comment='TD0_MPO_LOSS')

    # seeds for testing
    torch.manual_seed(seed)

    # environment parameters
    env, test_env = env_fn(), env_fn()
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    # this will slow down computation by 10-20 percent.
    # Only use for debugging
    # torch.autograd.set_detect_anomaly(True)

    # create actor-critic module and target networks
    if ac_kwargs is not dict():
        hid_q = ac_kwargs['hidden_sizes_q']
        hid_pi = ac_kwargs['hidden_sizes_pi']
    else:
        hid_q = (256, 256)
        hid_pi = (128, 128)
    ac = actor_critic(env,
                      hidden_sizes_q=hid_q,
                      hidden_sizes_pi=hid_pi)
    ac_targ = deepcopy(ac)

    # setting up lagrange values
    eta = torch.tensor([init_eta], requires_grad=True)
    eta_cov = init_eta_cov
    eta_mean = init_eta_mean

    # no update for target network with respect to optimizers (copied after k
    # optimizer steps)
    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # replay buffer
    replay_buffer = DynamicTrajectoryBuffer(s_dim,
                                            a_dim,
                                            max_ep_len,
                                            max_ep_len,
                                            len_rollout,
                                            5000,
                                            local_device)

    # counting variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)


    # setting up Adam Optimizer for gradient descent with momentum
    opti_q = Adam(q_params, lr=lr)
    # learn eta and policy parameters together
    opti_pi = Adam(itertools.chain(ac.pi.parameters(), [eta]), lr=lr)

    # set up logger to save model after each epoch

    def loss_q(run):
        rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_q)
        samples = replay_buffer.sample_batch(rows, cols)

        q1 = ac.q1(samples['state'], samples['action'])
        q2 = ac.q2(samples['state'], samples['action'])

        with torch.no_grad():
            targ_mean, targ_cov = ac_targ.pi.forward(samples['state_next'])
            next_act, _ = ac_targ.pi.get_act(targ_mean, targ_cov)
            q1_targ = ac_targ.q1(samples['state_next'], next_act)
            q2_targ = ac_targ.q2(samples['state_next'], next_act)
            q_targ = torch.min(q1_targ, q2_targ)
            backup = samples['reward'] + gamma * (1 - samples['done']) * q_targ

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        writer.add_scalar('loss_q', loss_q.item(), run)
        writer.add_scalar('q1_values', q1.detach().numpy().mean(), run)
        writer.add_scalar('q2_values', q2.detach().numpy().mean(), run)
        writer.add_scalar('q1_min', np.min(q1.detach().numpy()), run)
        writer.add_scalar('q1_max', np.min(q2.detach().numpy()), run)

        return loss_q

    def loss_q_retrace(run):
        rows, cols = replay_buffer.sample_idxs(batch_t)
        replay_traj = replay_buffer.sample_trajectories(rows, cols)


        curr_q_vals = ac.q1.forward(replay_traj['state'], replay_traj['action'])

        targ_q_vals = ac_targ.q1.forward(replay_traj['state'], replay_traj['action'])
        targ_mean, targ_cov = ac_targ.pi.forward(replay_traj['state'])
        pi_dist = Normal(targ_mean, torch.sqrt(targ_cov))
        targ_pol_prob = pi_dist.log_prob(replay_traj['action'])

        expected_q_vals = torch.zeros((len_rollout, batch_t), device=local_device)
        for i in range(len_rollout):
            targ_act, _ = ac_targ.pi.get_act(targ_mean[i, :, :],
                                             targ_cov[i, :, :])
            targ_act = targ_act.expand((batch_act, batch_t, a_dim))
            states = replay_traj['state'][i, :, :]
            states = states.expand((batch_act, batch_t, s_dim))
            expected_q_vals[i, :] = ac_targ.q1.forward(
                states, targ_act).mean(dim=0)

        retrace = Retrace()
        loss_q = retrace(curr_q_vals,
                         expected_q_vals,
                         targ_q_vals,
                         torch.squeeze(replay_traj['reward']),
                         torch.squeeze(targ_pol_prob),
                         torch.squeeze(replay_traj['pi_logp']),
                         gamma)
        writer.add_scalar('loss_q', loss_q, run)
        writer.add_scalar('q1_values', targ_q_vals.mean(), run)
        return loss_q


    def loss_pi(run):
        rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_s)
        samples = replay_buffer.sample_batch(rows, cols)

        targ_mean, targ_cov = ac_targ.pi.forward(samples['state'])
        exp_targ_mean = targ_mean.expand((batch_act, batch_s, a_dim))
        exp_targ_cov = targ_cov.expand((batch_act, batch_s, a_dim))
        targ_act_sampels, _ = ac_targ.pi.get_act(exp_targ_mean, exp_targ_cov)
        exp_state = samples['state'].expand((batch_act, batch_s, s_dim))

        targ_q_vals = torch.min(ac_targ.q1.forward(exp_state, targ_act_sampels),
                                ac_targ.q2.forward(exp_state, targ_act_sampels))

        q_weights = torch.softmax(targ_q_vals / eta, dim=0)
        cur_mean, cur_cov = ac.pi.forward(samples['state'])
        exp_cur_mean = cur_mean.expand((batch_act, batch_s, a_dim))
        exp_cur_cov = cur_cov.expand((batch_act, batch_s, a_dim))
        pi_dist_mean = Normal(exp_cur_mean, torch.sqrt(exp_targ_cov))
        pi_dist_cov = Normal(exp_targ_mean, torch.sqrt(exp_cur_cov))

        loss_p = torch.mean(
            q_weights.unsqueeze(dim=1) * (
                pi_dist_mean.log_prob(targ_act_sampels) +
                pi_dist_cov.log_prob(targ_act_sampels)
            )
        )
        writer.add_scalar('loss_pi', -loss_p.item(), run)

        c_mean, c_cov = compute_lagr_loss(cur_mean, cur_cov, targ_mean, targ_cov)

        loss_eta_mean = alpha*(eps_mean - c_mean.detach()).item()
        writer.add_scalar('loss_eta_mean', loss_eta_mean, run)
        nonlocal eta_mean
        eta_mean -= loss_eta_mean
        loss_eta_cov = alpha*(eps_cov - c_cov.detach()).item()
        writer.add_scalar('loss_eta_cov', loss_eta_cov, run)
        nonlocal eta_cov
        eta_cov -= loss_eta_cov

        if eta_mean < 0:
            eta_mean = 0
        if eta_cov < 0:
            eta_cov = 0

        writer.add_scalar('eta_mean', eta_mean, run)
        writer.add_scalar('eta_cov', eta_cov, run)

        # update eta
        max_q = torch.max(targ_q_vals, dim=0).values
        inner = targ_q_vals.squeeze() - max_q
        loss_eta = eps * eta + max_q.mean() + eta * torch.mean(torch.log(torch.mean(
            torch.exp(inner / eta), dim=0)))

        writer.add_scalar('loss_eta', loss_eta.item(), run)
        writer.add_scalar('eta', eta.item(), run)

        return -(
            loss_p + eta_mean * (eps_mean - c_mean) + eta_cov * (eps_cov - c_cov)
        ) + loss_eta

    def loss_pi_sac(run):
        rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_s)
        samples = replay_buffer.sample_batch(rows, cols)
        cur_mean, cur_cov = ac.pi.forward(samples['state'])
        cur_act, logp_pi = ac.pi.get_act(cur_mean, cur_cov)

        with torch.no_grad():
            q1_pi = ac.q1(samples['state'], cur_act)
            q2_pi = ac.q(samples['state'], cur_act)
            q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (0.1 * logp_pi - q_pi).mean()
        writer.add_scalar('loss_pi', loss_pi, run)
        writer.add_scalar('logp_pi', logp_pi.detach().numpy.mean(), run)
        return loss_pi

    def compute_lagr_loss(cur_mean, cur_cov, targ_mean, targ_cov):
        # dimensions to be (batch_s, a_dim)
        n = a_dim
        combined_trace = ((1 / cur_cov) * targ_cov).sum(dim=1)
        target_det = targ_cov.prod(dim=1)
        current_det = cur_cov.prod(dim=1)
        log_det = (current_det / target_det).log()
        c_mean = 0.5 * (combined_trace - n + log_det).mean()

        dif = cur_mean - targ_mean
        c_cov = 0.5 * (((dif ** 2) * (1 / cur_cov)).sum(dim=1)).mean()

        return c_mean, c_cov

    def get_action(state, deterministic=False):
        action, logp_pi = ac.act(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=local_device),
            deterministic=deterministic)
        return action, logp_pi

    def test_agent(run):
        for j in tqdm(range(num_test_episodes), desc="testing model"):
            s, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                with torch.no_grad():
                    s, r, d, _ = test_env.step(
                        get_action(s, deterministic=True)[0])
                ep_ret += r
                ep_len += 1
            writer.add_scalar('episode_return', ep_ret, run * num_test_episodes + j)
            writer.add_scalar('episode_length', ep_len, run * num_test_episodes + j)

    def sample_traj(perform_traj=1, random_act=False):
        for _ in range(perform_traj):

            # sample steps
            s, ep_ret, ep_len = env.reset(), 0, 0
            while True:
                # sample from random policy for better exploration
                if random_act:
                    a = env.action_space.sample()
                    mu_rand, cov_rand = ac_targ.pi.forward(
                        torch.as_tensor(s, dtype=torch.float32, device=local_device))
                    logp = ac_targ.pi.get_logp(
                        mu_rand, cov_rand, torch.as_tensor(a, dtype=torch.float32, device=local_device))
                else:
                    # action from current policy
                    a, logp = get_action(s)
                # do step in environment
                s2, r, d, _ = env.step(a)
                r = reward_scaling(r)
                ep_ret += r
                ep_len += 1

                d = False if ep_len == max_ep_len else d

                replay_buffer.store(s.reshape(s_dim), s2.reshape(s_dim), a, r, logp.cpu().numpy(), d)
                s = s2
                # reset environment (ignore done signal)
                if ep_len == max_ep_len:
                    s, ep_ret, ep_len = env.reset(), 0, 0
                    replay_buffer.next_traj()
                    break

    start_time = time.time()

    # main loop
    performed_trajectories = 0
    for i in range(epochs):
        for t in tqdm(range(traj_update_count), desc='sample and update Q'):
            sample_traj(perform_traj=1)
            performed_trajectories += traj_update_count

            for run in range(runs_update_q):
                opti_q.zero_grad()

                loss = loss_q(i*runs_update_q*traj_update_count + run + t*runs_update_q)
                # loss = loss_q_retrace(i*runs_update_q + run)

                loss.backward()

                opti_q.step()
                with torch.no_grad():
                    for p, p_targ in zip(ac.q1.parameters(), ac_targ.q1.parameters()):
                        p_targ.data.mul_(polyak)
                    for p, p_targ in zip(ac.q2.parameters(), ac_targ.q2.parameters()):
                        p_targ.data.add((1-polyak)*p.data)

        for run in tqdm(range(runs_update_pi), desc='update policy'):
            opti_pi.zero_grad()

            loss = loss_pi(i * runs_update_pi + run)
            loss.backward()

            opti_pi.step()

            if eta.item() < 0:
                eta -= eta
        with torch.no_grad():
            for p, p_targ in zip(ac.pi.parameters(), ac_targ.pi.parameters()):
                p_targ.data.copy_(p.data)

        test_agent(i)
        writer.add_scalar('time_per_epoch', time.time() - start_time, i)
        start_time = time.time()
        writer.flush()


