from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
import torch
from torch.optim import Adam
import gym
import time
from mpo_algorithm import core
from utils.logx import EpochLogger
from mpo_algorithm.retrace import Retrace
from mpo_algorithm.tray_dyn_buf import DynamicTrajectoryBuffer
from tqdm import tqdm
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent

local_device = "cpu"


def mpo(env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        replay_size=int(1e6),
        gamma=0.99,
        epochs=2000,
        traj_update_count=20,
        max_ep_len=1000,
        eps=0.1,
        eps_mean=0.1,
        eps_cov=0.0001,
        lr=0.0005,
        lr_lagr=0.0005,
        alpha=0.1,
        batch_t=10,
        batch_s=768,
        batch_act=20,
        len_rollout=1000,
        init_eta=1.0,
        init_eta_mean=1.0,
        init_eta_cov=1.0,
        update_target_after=1000,
        num_test_episodes=200,
        reward_scaling=lambda r: r,
        logger_kwargs=dict()):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    max_traj = epochs * traj_update_count

    # seeds for testing
    # torch.manual_seed(seed)
    # np.random.seed(seed)

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

    # setting up lagrange values where gradient is required
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
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    def compute_lagr_loss(cur_mean, cur_cov, targ_mean, targ_cov):
        # dimensions to be (batch_s, a_dim)
        n = a_dim
        combined_trace = ((1 / cur_cov) * targ_cov).sum(dim=1)
        target_det = targ_cov.prod(dim=1)
        current_det = cur_cov.prod(dim=1)
        log_det = (current_det / target_det).log()
        c_mean = 0.5 * (combined_trace - n + log_det).mean()

        dif = targ_mean - cur_mean
        c_cov = 0.5 * (((dif ** 2) * (1 / cur_cov)).sum(dim=1)).mean()

        return c_mean, c_cov

    def compute_pi_loss(targ_q_vals,  # (batch_act, batch_s)
                        cur_mean,  # (batch_s,)
                        cur_cov,
                        targ_mean,
                        targ_cov,
                        c_mean,
                        c_cov,
                        samples_act  # (batch_act, batch_s, a_dim)
                        ):

        dist_1 = Independent(Normal(cur_mean, targ_cov), 1)
        dist_2 = Independent(Normal(targ_mean, cur_cov), 1)

        q_weights = torch.softmax(targ_q_vals / eta, dim=0)
        loss = torch.mean(
            q_weights * (
                    dist_1.expand((batch_act, batch_s)).log_prob(samples_act) +
                    dist_2.expand((batch_act, batch_s)).log_prob(samples_act)
            ))
        combined_loss = -(loss + eta_mean * (eps_mean - c_mean) + eta_cov * (eps_cov - c_cov))
        logger.store(LossPi=combined_loss.item())
        return combined_loss

    def compute_pi_loss2(targ_q_vals,  # (batch_act, batch_s)
                        cur_mean,  # (batch_s,)
                        cur_cov,
                        targ_mean,
                        targ_cov,
                        c_mean,
                        c_cov,
                        samples_act  # (batch_act, batch_s, a_dim)
                        ):

        cur_dist = Independent(Normal(cur_mean, cur_cov), 1)

        q_weights = torch.softmax(targ_q_vals / eta, dim=0)
        samples_act_weighted = q_weights.unsqueeze(1) * samples_act
        # todo add clip
        loss = torch.mean(
                    cur_dist.expand((batch_act, batch_s)).log_prob(samples_act_weighted)
            )
        combined_loss = -(loss + eta_mean * (eps_mean - c_mean) + eta_cov * (eps_cov - c_cov))
        logger.store(LossPi=combined_loss.item())
        return combined_loss


    # setting up Adam Optimizer for gradient descent with momentum
    opti_q = Adam(ac.q.parameters(), lr=lr)
    opti_pi = Adam(ac.pi.parameters(), lr=lr)

    # set up logger to save model after each epoch
    logger.setup_pytorch_saver(ac)

    def update_eta(targ_q_vals):
        # update via dual function
        nonlocal eta

        def compute_eta_loss(eta):
            # q_values: (batch_act, batch_s)
            # use max baseline for numerical stability
            max_q = torch.max(targ_q_vals, dim=0).values.cpu().data.numpy()
            inner = targ_q_vals.squeeze().cpu().data.numpy() - max_q
            return eps * eta + np.mean(max_q) + eta * np.mean(np.log(np.mean(
                np.exp(inner / eta), axis=0)))

        res = minimize(
            compute_eta_loss, np.array([eta]), method='SLSQP', bounds=[(1e-6, None)])
        eta = res.x[0]
        eta = max(1e-8, eta)
        logger.store(Eta=eta)

    def update_eta_lagr(cur_mean, cur_cov, targ_mean, targ_cov):
        # update for lagrange values with simple gradient descent
        nonlocal eta_mean
        nonlocal eta_cov
        c_mean, c_cov = compute_lagr_loss(cur_mean, cur_cov, targ_mean, targ_cov)

        loss_eta_mean = lr_lagr * (eps_mean - c_mean).detach().item()
        eta_mean -= loss_eta_mean

        loss_eta_cov = lr_lagr * (eps_cov - c_cov).detach().item()
        eta_cov -= loss_eta_cov

        eta_mean = max(1e-8, eta_mean)
        eta_cov = max(1e-8, eta_cov)

        logger.store(EtaMean=eta_mean)
        logger.store(LossEtaMean=loss_eta_mean)
        logger.store(EtaCov=eta_cov)
        logger.store(LossEtaCov=loss_eta_cov)

        return c_mean, c_cov

    def update_q(rows, cols):
        opti_q.zero_grad()

        sample_traj = replay_buffer.sample_trajectories(rows, cols)
        curr_q_vals = ac.q.forward(sample_traj['state'], sample_traj['action'])
        targ_q_vals = ac_targ.q.forward(sample_traj['state'], sample_traj['action'])
        targ_mean, targ_cov = ac_targ.pi.forward(sample_traj['state'])
        targ_pol_prob = ac_targ.pi.get_dist(
            targ_mean, targ_cov).log_prob(sample_traj['action'])

        expected_q_vals = torch.zeros((len_rollout, batch_t), device=local_device)
        for i in range(len_rollout):
            targ_act, _ = ac_targ.pi.get_act(targ_mean[i, :, :],
                                             targ_cov[i, :, :],
                                             batch_act)
            states = sample_traj['state'][i, :, :]
            states = states.expand((batch_act, batch_t, s_dim))
            expected_q_vals[i, :] = ac_targ.q.forward(
                states, targ_act).mean(dim=0)

        retrace = Retrace()
        loss_q = retrace(curr_q_vals,
                         expected_q_vals,
                         targ_q_vals,
                         torch.squeeze(sample_traj['reward']),
                         torch.squeeze(targ_pol_prob),
                         torch.squeeze(sample_traj['pi_logp']),
                         gamma)

        loss_q.backward()
        opti_q.step()
        logger.store(LossQ=loss_q.item())

    def prep_data(rows, cols):
        samples = replay_buffer.sample_batch(rows, cols)
        states = samples['state']

        targ_mean, targ_cov = ac_targ.pi.forward(states)
        cur_mean, cur_cov = ac.pi.forward(states)
        targ_act_samples, _ = ac_targ.pi.get_act(targ_mean,
                                                 targ_cov,
                                                 batch_act)

        states = states.expand((batch_act, batch_s, s_dim))
        targ_q_vals = ac_targ.q.forward(states, targ_act_samples)

        return targ_q_vals, cur_mean, cur_cov, targ_mean, targ_cov, targ_act_samples

    def get_action(state, deterministic=False):
        action, logp_pi = ac.act(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=local_device),
            deterministic=deterministic)
        return action, logp_pi

    def test_agent():
        for j in tqdm(range(num_test_episodes), desc="testing model"):
            s, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                with torch.no_grad():
                    s, r, d, _ = test_env.step(
                        get_action(s, deterministic=True)[0])
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    s, ep_ret, ep_len = env.reset(), 0, 0

    # main loop
    performed_trajectories = 0
    epoch = 0
    first_random_traj = 30
    while performed_trajectories < max_traj:

        # sample trajectories from environment
        # and safe in replay buffer
        for _ in tqdm(range(traj_update_count), desc='sampling trajectories'):

            # sample steps
            while True:
                # sample from random policy for better exploration
                if performed_trajectories < first_random_traj:
                    a = env.action_space.sample()
                    mu_rand, cov_rand = ac_targ.pi.forward(
                        torch.as_tensor(s, dtype=torch.float32, device=local_device))
                    logp = ac_targ.pi.get_dist(
                        mu_rand, cov_rand).log_prob(
                        torch.as_tensor(a, dtype=torch.float32, device=local_device))
                else:
                    # action from current policy
                    a, logp = get_action(s)
                # do step in environment
                s2, r, d, _ = env.step(a)
                r = reward_scaling(r)
                ep_ret += r
                ep_len += 1

                # d = False if ep_len == max_ep_len else d

                replay_buffer.store(s.reshape(s_dim), a, r, logp.cpu().numpy())
                s = s2
                # reset environment (ignore done signal)
                if ep_len == max_ep_len:
                    logger.store(OneStepRet=ep_ret/ep_len)
                    s, ep_ret, ep_len = env.reset(), 0, 0
                    replay_buffer.next_traj()
                    break

        performed_trajectories += traj_update_count

        # update q function
        rows, cols = replay_buffer.sample_idxs(batch_size=batch_t)
        update_q(rows, cols)

        # update eta by dual optimization
        for _ in range(10):
            rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_t * batch_s)
            samples = replay_buffer.sample_batch(rows, cols)
            states = samples['state']

            targ_mean, targ_cov = ac_targ.pi.forward(states)
            cur_mean, cur_cov = ac.pi.forward(states)
            targ_act_samples, _ = ac_targ.pi.get_act(targ_mean,
                                                     targ_cov,
                                                     batch_act)

            states = states.expand((batch_act, batch_t * batch_s, s_dim))
            targ_q_vals = ac_targ.q.forward(states, targ_act_samples)
            update_eta(targ_q_vals)
        for k in tqdm(range(update_target_after), desc='updating policy'):
            # sample random batch
            rows, cols = replay_buffer.sample_idxs_batch(batch_size=batch_s)
            # perform gradient descent step
            targ_q_vals, cur_mean, cur_cov, targ_mean, targ_cov, act_samples = \
                prep_data(rows, cols)

            # log q and actor values
            logger.store(QVal=targ_q_vals[0][0].item())
            # actor values can only be logged for pendulum !!!
            logger.store(ActorMean=cur_mean[0][0].item())
            logger.store(ActorCov=cur_cov[0][0].item())

            c_mean, c_cov = update_eta_lagr(cur_mean,
                                            cur_cov,
                                            targ_mean,
                                            targ_cov)
            # updating pi
            opti_pi.zero_grad()
            loss_pi = compute_pi_loss(
                targ_q_vals,
                cur_mean,
                cur_cov,
                targ_mean,
                targ_cov,
                c_mean,
                c_cov,
                act_samples
            )
            loss_pi.backward()
            opti_pi.step()

        # update target parameters
        # use ugly hack until better option is found
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(0)
                p_targ.data.add_(p)

        # after each epoch evaluate performance of agent
        epoch += 1
        logger.save_state({'env': env}, None)
        test_agent()

        # Log all relevant information
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Performed Trajectories', performed_trajectories)
        logger.log_tabular('Environment Interactions',
                           replay_buffer.stored_interactions())
        logger.log_tabular('TestEpRet', with_min_and_max=True)

        logger.log_tabular('LossQ', with_min_and_max=False)
        logger.log_tabular('LossPi', with_min_and_max=False)
        logger.log_tabular('LossEtaCov', with_min_and_max=False)
        logger.log_tabular('LossEtaMean', with_min_and_max=False)

        logger.log_tabular('EtaMean', with_min_and_max=False)
        logger.log_tabular('EtaCov', with_min_and_max=False)
        logger.log_tabular('Eta', with_min_and_max=False)
        logger.log_tabular('QVal', with_min_and_max=True)
        logger.log_tabular('OneStepRet', with_min_and_max=True)
        logger.log_tabular('ActorMean', with_min_and_max=True)
        logger.log_tabular('ActorCov', with_min_and_max=True)

        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        start_time = time.time()

