from copy import deepcopy
import numpy as np
import itertools
import torch
from torch.distributions import Independent
from torch.distributions.normal import Normal
from torch.optim import Adam
import time
from mpo_retrace import core
from torch.utils.tensorboard import SummaryWriter
from mpo_retrace.tray_dyn_buf import DynamicTrajectoryBuffer
from tqdm import tqdm
from mpo_retrace.retrace import Retrace
from scipy.optimize import minimize

local_device = "cpu"


def mpo_retrace(env_fn,
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
                lr_pi=5e-4,
                lr_q=2e-4,
                alpha=10,
                batch_t=1,  # sampled trajectories per learning step
                batch_s=200,
                batch_eta=4096,
                batch_act=20,  # additional samples for integral estimation
                len_rollout=200,
                init_eta=1.,
                init_eta_mean=10.0,
                init_eta_cov=10.0,
                learning_steps=1000,
                update_targ_nets_after=200,
                num_test_episodes=200,
                polyak=0.995,
                reward_scaling=lambda r: r):
    writer = SummaryWriter(comment='MPO_RETRACE_ALPHA_10')

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
        hid_pi = (256, 256)
    ac = actor_critic(env,
                      hidden_sizes_q=hid_q,
                      hidden_sizes_pi=hid_pi)
    ac_targ = deepcopy(ac)

    # setting up lagrange values
    eta = init_eta
    eta_cov = init_eta_cov
    eta_mean = init_eta_mean

    # no update for target network with respect to optimizers (copied after k
    # optimizer steps)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # replay buffer
    replay_buffer = DynamicTrajectoryBuffer(s_dim,
                                            a_dim,
                                            max_ep_len,
                                            max_ep_len,
                                            len_rollout,
                                            5000,
                                            local_device)

    # counting variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d\n' % var_counts)

    # setting up Adam Optimizer for gradient descent with momentum
    opti_q = Adam(ac.q.parameters(), lr=lr_q)
    # learn eta and policy parameters together
    opti_pi = Adam(ac.pi.parameters(), lr=lr_pi)

    def loss_q_retrace(samples, run):
        batch_q = ac.q.forward(samples['state'], samples['action'])
        batch_q = torch.transpose(batch_q, 0, 1)

        targ_q = ac_targ.q.forward(samples['state'], samples['action'])
        targ_q = torch.transpose(targ_q, 0, 1)

        targ_mean, targ_std = ac_targ.pi.forward(samples['state'])
        targ_act, _ = ac_targ.pi.get_act(targ_mean, targ_std)

        exp_targ_q = ac_targ.q.forward(samples['state'], targ_act)
        exp_targ_q = torch.transpose(exp_targ_q, 0, 1)

        targ_act_logp = ac_targ.pi.get_logp(targ_mean, targ_std, samples['action'])
        targ_act_logp = torch.transpose(targ_act_logp, 0, 1)

        retrace = Retrace()
        loss_q = retrace(Q=batch_q,
                         expected_target_Q=exp_targ_q,
                         target_Q=targ_q,
                         rewards=torch.transpose(samples['reward'], 0, 1).squeeze(-1),
                         target_policy_probs=targ_act_logp,
                         behaviour_policy_probs=torch.transpose(samples['pi_logp'], 0, 1).squeeze(-1),
                         gamma=gamma
                         )
        writer.add_scalar('q_loss', loss_q.item(), run)
        writer.add_scalar('q', targ_q.detach().numpy().mean(), run)
        writer.add_scalar('q_min', targ_q.detach().numpy().min(), run)
        writer.add_scalar('q_max', targ_q.detach().numpy().max(), run)

        return loss_q

    def compute_lagr_loss(cur_mean, cur_std, targ_mean, targ_std):
        # dimensions to be (batch_s, a_dim)
        n = a_dim
        cur_cov = cur_std ** 2
        targ_cov = targ_std ** 2

        combined_trace = ((1 / cur_cov) * targ_cov).sum(dim=1)
        target_det = targ_cov.prod(dim=1)
        current_det = cur_cov.prod(dim=1)
        log_det = (current_det / target_det).log()
        c_mean = 0.5 * (combined_trace - n + log_det).mean()

        dif = cur_mean - targ_mean
        c_cov = 0.5 * (((dif ** 2) * (1 / cur_cov)).sum(dim=1)).mean()
        return c_mean, c_cov

    def loss_pi_mpo(samples, run):
        targ_mean, targ_std = ac_targ.pi.forward(samples['state'])
        exp_targ_mean = targ_mean.expand((batch_act, batch_s, a_dim))
        exp_targ_std = targ_std.expand((batch_act, batch_s, a_dim))
        targ_act, targ_logp = ac_targ.pi.get_act(exp_targ_mean, exp_targ_std)  # get additional action samples

        writer.add_scalar('pi_logp', targ_logp.detach().numpy().mean(), run)

        exp_state = samples['state'].expand((batch_act, batch_s, s_dim))
        targ_q = ac_targ.q.forward(exp_state, targ_act)

        q_weights = torch.softmax(targ_q / eta, dim=0)
        cur_mean, cur_std = ac.pi.forward(samples['state'])
        pi_dist_1 = Independent(Normal(cur_mean, targ_std), reinterpreted_batch_ndims=a_dim)
        pi_dist_2 = Independent(Normal(targ_mean, cur_std), reinterpreted_batch_ndims=a_dim)

        loss_p = torch.mean(
            q_weights * (
                    pi_dist_1.expand((batch_act, batch_s)).log_prob(targ_act) +
                    pi_dist_2.expand((batch_act, batch_s)).log_prob(targ_act)
            )
        )
        writer.add_scalar('pi_loss', -(loss_p.detach().numpy().mean()), run)

        c_mean, c_cov = compute_lagr_loss(cur_mean, cur_std, targ_mean, targ_std)
        writer.add_scalar('c_mean', c_mean.detach().numpy().mean(), run)
        writer.add_scalar('c_cov', c_cov.detach().numpy().mean(), run)

        loss_eta_mean = alpha * (eps_mean - c_mean.detach()).item()
        writer.add_scalar('eta_mean_loss', loss_eta_mean, run)
        nonlocal eta_mean
        eta_mean -= loss_eta_mean

        loss_eta_cov = alpha * (eps_cov - c_cov.detach()).item()
        writer.add_scalar('eta_cov_loss', loss_eta_cov, run)
        nonlocal eta_cov
        eta_cov -= loss_eta_cov

        if eta_mean < 0:
            eta_mean = 0
        if eta_cov < 0:
            eta_cov = 0

        writer.add_scalar('eta_mean', eta_mean, run)
        writer.add_scalar('eta_cov', eta_cov, run)

        loss_combined = -(loss_p + eta_mean * (eps_mean - c_mean) + eta_cov * (eps_cov - c_cov))
        writer.add_scalar('combined_loss', loss_combined.detach().numpy().mean(), run)

        writer.add_scalar('eta', eta, run)

        return loss_combined

    def update_eta(samples, run):
        targ_mean, targ_std = ac_targ.pi.forward(samples['state'])
        exp_targ_mean = targ_mean.expand((batch_act, batch_eta, a_dim))
        exp_targ_std = targ_std.expand((batch_act, batch_eta, a_dim))
        targ_act, targ_logp = ac_targ.pi.get_act(exp_targ_mean, exp_targ_std)  # get additional action samples

        exp_state = samples['state'].expand((batch_act, batch_eta, s_dim))
        targ_q = ac_targ.q.forward(exp_state, targ_act).cpu().numpy()

        def dual(eta):
            max_q = np.max(targ_q, axis=0)
            return eta * eps + np.mean(max_q) + eta * np.mean(
                np.log(np.mean(np.exp((targ_q - max_q)/ eta), axis=0))
            )
        bounds = [(1e-6, None)]
        nonlocal eta
        res = minimize(dual, np.array(eta), method='SLSQP', bounds=bounds)
        eta = res.x[0]
        writer.add_scalar('eta', eta, run)

    def get_action(state, deterministic=False):
        action, logp_pi = ac.act(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=local_device),
            deterministic=deterministic)
        return action, logp_pi

    def test_agent(run):
        ep_ret_list = list()
        for j in tqdm(range(num_test_episodes), desc="testing model"):
            s, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                with torch.no_grad():
                    s, r, d, _ = test_env.step(
                        get_action(s, deterministic=True)[0])
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
        writer.add_scalar('test_ep_ret', np.array(ep_ret_list).mean(), run)

    def sample_traj(perform_traj=1, random_act=False):
        for _ in range(perform_traj):

            # sample steps
            s, ep_ret, ep_len = env.reset(), 0, 0
            while True:
                # sample from random policy for better exploration
                if random_act:
                    a = env.action_space.sample()
                    mu_rand, std_rand = ac_targ.pi.forward(
                        torch.as_tensor(s, dtype=torch.float32, device=local_device))
                    logp = ac_targ.pi.get_logp(
                        mu_rand, std_rand, torch.as_tensor(a, dtype=torch.float32, device=local_device))
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
    for i in range(epochs):
        # sample traj_update_count many trajectories to update replay buffer

        for j in tqdm(range(learning_steps), desc='update nets'):

            if j % (learning_steps // traj_update_count) == 0:
                if i == 0:
                    sample_traj(perform_traj=10)
                else:
                    sample_traj(perform_traj=1)

                if i > 0:
                    rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_eta)
                    samples = replay_buffer.sample_batch(rows, cols)
                    update_eta(samples, run=i * learning_steps + j)

            if j % update_targ_nets_after == 0:
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.copy_(p.data)

            # update q
            rows, cols = replay_buffer.sample_idxs(batch_size=batch_t)
            for _ in range(10):
                samples = replay_buffer.sample_trajectories(rows, cols)
                opti_q.zero_grad()
                loss = loss_q_retrace(samples, run=i * learning_steps + j)
                loss.backward()
                opti_q.step()

            # update pi
            rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_s)
            for _ in range(10):
                samples = replay_buffer.sample_batch(rows, cols)

                opti_pi.zero_grad()
                loss = loss_pi_mpo(samples, run=i * learning_steps + j)
                loss.backward()
                opti_pi.step()

        test_agent(i)
        writer.add_scalar('time_per_epoch', time.time() - start_time, i)
        start_time = time.time()
        writer.flush()
