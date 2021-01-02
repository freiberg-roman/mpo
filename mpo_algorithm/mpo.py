from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from mpo_algorithm import core
from utils.logx import EpochLogger
from mpo_algorithm.retrace import Retrace
from mpo_algorithm.tray_dyn_buf import DynamicTrajectoryBuffer

local_device = "cuda:0"



def mpo(env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(),
        seed=0,
        replay_size=int(1e6),
        gamma=0.99,
        epochs=2000,
        traj_update_count=20,
        max_ep_len=200,
        eps=0.1,
        eps_mu=0.1,
        eps_sig=0.0001,
        lr=0.0005,
        alpha=0.1,
        batch_size_replay=128,
        batch_size_policy=64,
        init_eta=10.0,
        init_eta_mu=10.0,
        init_eta_sigma=10.0,
        update_target_after=1000,
        num_test_episodes=50,
        action_dim=1,
        logger_kwargs=dict()):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    max_traj = epochs * traj_update_count

    # seeds for testing
    torch.manual_seed(seed)
    np.random.seed(seed)

    # environment parameters
    env, test_env = env_fn(), env_fn()
    state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    action_dim = action_dim

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
    ac = actor_critic(env.observation_space,
                      action_dim,
                      hidden_sizes_q=hid_q,
                      hidden_sizes_pi=hid_pi)
    ac_targ = deepcopy(ac)

    # setting up lagrange values where gradient is required
    eta = torch.tensor([init_eta],
                       requires_grad=True,
                       device=local_device)
    eta_sig = torch.tensor([init_eta_sigma],
                           requires_grad=True,
                           device=local_device)
    eta_mu = torch.tensor([init_eta_mu],
                          requires_grad=True,
                          device=local_device)

    # no update for target network with respect to optimizers (copied after k
    # optimizer steps)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # replay buffer
    replay_buffer = DynamicTrajectoryBuffer(state_dim,
                                            action_dim,
                                            10,
                                            max_ep_len,
                                            5000,
                                            local_device)

    # counting variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    def compute_loss_q(data):
        # retrace implementation provided by denis bless
        q = torch.reshape(data['q'], (1, batch_size_replay))
        # 𝔼_π Q(s_t,.) is estimated with values samples by policy i.e
        # 1/batch_size_policy * sum(Q(s_t,a_pi)) over a_pi in batch
        # where Q is the current Q function
        # note: the states are the same as used for q_values
        expected_q = torch.reshape(data['expected_q'], (1, batch_size_replay))
        # values used from delayed network (target network)
        q_target = torch.reshape(data['targ_q'], (1, batch_size_replay))
        rew = torch.reshape(data['rew'], (1, batch_size_replay))
        # π(a|s_t) values from current policy
        # note: the states are the same as used for q_values
        target_pi_prob = torch.reshape(data['pi'], (1, batch_size_replay))
        # behaviour policy values b(a|s_t) are emulated by using old values
        # in replay buffer
        behaviour_policy_prob = torch.reshape(data['b'], (1, batch_size_replay))

        retrace = Retrace()
        return retrace(q,
                       expected_q,
                       q_target,
                       rew,
                       target_pi_prob,
                       behaviour_policy_prob,
                       gamma)

    def compute_eta(q_values):
        # q_values: tensor of shape (batch_size_replay, batch_size_policy)
        return eps * eta + eta * torch.mean(torch.log(torch.mean(
            torch.exp(q_values / eta), 1
        )))

    def compute_lagr_loss(eta_mu, eta_sig, data):
        # assume that all target_sigma_values and current_sigma_values are of shape
        # [B, [n,]] where B is the batch size for integral estimation and [n,] is the
        # diagonal covariance matrices
        target_sigma = data['tar_sig']
        current_sigma = data['cur_sig']
        current_mu = data['cur_mu']
        target_mu = data['tar_mu']

        n = target_sigma.shape[1]
        combined_trace = torch.sum((1 / current_sigma) * target_sigma, dim=1)
        target_det = torch.prod(target_sigma, dim=1)
        current_det = torch.prod(current_sigma, dim=1)
        log_det = torch.log(current_det / target_det)
        c_mu = 0.5 * torch.mean(combined_trace - n + log_det)
        lagr_eta_mu = eta_mu * (eps_mu - c_mu)

        dif = target_mu - current_mu
        c_sig = 0.5 * torch.mean(torch.sum((dif ** 2) * current_sigma, dim=-1))
        lagr_eta_sig = eta_sig * (eps_sig - c_sig)
        res = lagr_eta_mu + lagr_eta_sig
        return res

    def compute_pi_loss(data, lagr):
        # target_q: tensor of shape (batch_size_replay, batch_size_policy)
        target_q = torch.squeeze(data['targ_q_batch'])
        # cur_logp: tensor of shape (batch_size_replay, batch_size_policy)
        cur_logp = torch.squeeze(data['cur_logp'], dim=-1)
        eta.requires_grad = False
        exp_q_eta = torch.exp(target_q / eta)
        res = torch.mean(torch.mean(cur_logp * exp_q_eta, dim=-1), dim=-1)
        eta.requires_grad = True
        return -(res + lagr)

    # setting up Adam Optimizer for gradient descent with momentum
    q_optimizer = Adam(ac.q.parameters(), lr=lr)
    eta_optimizer = Adam([eta], lr=lr)
    eta_lagr_optimizer = Adam([eta_mu, eta_sig], lr=lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)

    logger.setup_pytorch_saver(ac)

    def update(data):
        # update for eta
        eta_optimizer.zero_grad()
        eta_data = data['eta']
        loss_eta = compute_eta(eta_data)
        loss_eta.backward()
        eta_optimizer.step()

        # update for policies
        pi_optimizer.zero_grad()
        cur_mu, cur_cov = ac.pi.forward(data['states'])

        # estimating policy loss
        cur_act_samples, cur_logp = ac.pi.get_act(cur_mu,
                                                  cur_cov,
                                                  batch_size_policy)
        cur_logp = torch.reshape(cur_logp,
                                 (batch_size_replay, batch_size_policy, 1))

        # update for lagrange values
        eta_lagr_optimizer.zero_grad()
        eta_lagr_data_no_grad = data['eta_lagr_no_grad']
        eta_lagr_data_no_grad['cur_mu'] = cur_mu.clone().detach()
        eta_lagr_data_no_grad['cur_sig'] = cur_cov.clone().detach()
        loss_eta_lagr = compute_lagr_loss(
            eta_mu, eta_sig, eta_lagr_data_no_grad)
        loss_eta_lagr.backward()
        eta_lagr_optimizer.step()

        pi_data = data['pi']
        pi_data['cur_sig'] = cur_cov
        pi_data['cur_mu'] = cur_mu
        pi_data['cur_logp'] = cur_logp
        eta_lagr_data = data['eta_lagr']
        eta_lagr_data['cur_mu'] = cur_mu
        eta_lagr_data['cur_sig'] = cur_cov
        loss_eta_lagr = compute_lagr_loss(
            eta_mu.clone().detach(),
            eta_sig.clone().detach(),
            data['eta_lagr']
        )
        loss_pi = compute_pi_loss(pi_data, loss_eta_lagr)
        loss_pi.backward()
        pi_optimizer.step()

        # logger
        logger.store(LossEta=loss_eta.item(), Eta=eta.item())
        logger.store(LossPi=loss_pi.item())
        logger.store(LossLagr=loss_eta_lagr.item())
        logger.store(EtaMu=eta_mu.item())
        logger.store(EtaSig=eta_sig.item())

    def update_q():
        q_optimizer.zero_grad()

        traj_batch_size = 2
        sample_traj = replay_buffer.sample_traj(traj_batch_size=traj_batch_size)
        traj_len = sample_traj['state'].size()[1]
        curr_q_vals = ac.q.forward(sample_traj['state'], sample_traj['action'])
        targ_q_vals = ac_targ.q.forward(sample_traj['state'], sample_traj['action'])
        mu_targ, cov_targ = ac_targ.pi.forward(sample_traj['state'])
        targ_pol_prob = ac_targ.pi.get_prob(mu_targ, cov_targ, sample_traj['action'])

        expected_q_vals = torch.zeros((traj_batch_size, traj_len), device=local_device)
        for i in range(traj_batch_size):
            targ_actions, _ = ac_targ.pi.get_act(mu_targ[i, :, :],
                                                 cov_targ[i, :, :],
                                                 batch_size_policy)
            targ_actions = torch.reshape(
                targ_actions, (batch_size_policy, traj_len, action_dim))
            states = sample_traj['state'][i, :, :]
            states = states.repeat(batch_size_policy, 1, 1)
            expected_q_vals[i, :] = torch.mean(
                ac_targ.q.forward(states, targ_actions), dim=0)

        retrace = Retrace()
        loss_q = retrace(curr_q_vals,
                         expected_q_vals,
                         targ_q_vals,
                         torch.squeeze(sample_traj['reward']),
                         torch.squeeze(targ_pol_prob),
                         torch.squeeze(sample_traj['pi_logp']),
                         gamma)

        loss_q.backward()
        q_optimizer.step()
        logger.store(LossQ=loss_q.item())

    def prep_data():
        samples = replay_buffer.sample_batch(batch_size=batch_size_replay)
        states = samples['state']

        data = dict()
        data['states'] = states

        targ_mu, targ_cov = ac_targ.pi.forward(states)
        targ_act_samples, _ = ac_targ.pi.get_act(targ_mu,
                                                 targ_cov,
                                                 batch_size_policy)

        # get Q values for samples from current Q function
        # get Q values from target Q function
        states = states.repeat(batch_size_policy, 1)
        targ_q_batch = ac_targ.q.forward(states, targ_act_samples)
        targ_q_batch = torch.reshape(targ_q_batch, (batch_size_replay, batch_size_policy, 1))

        data['eta'] = targ_q_batch
        data['eta_lagr_no_grad'] = dict(
            tar_mu=targ_mu,
            tar_sig=targ_cov,
        )
        data['eta_lagr'] = dict(
            tar_mu=targ_mu,
            tar_sig=targ_cov,
        )
        data['pi'] = dict(
            targ_q_batch=targ_q_batch,
        )

        return data

    def get_action(state, deterministic=False):
        action, logp_pi = ac.act(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=local_device),
            deterministic=deterministic)
        return action, logp_pi

    def test_agent():
        for j in range(num_test_episodes):
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
    while performed_trajectories < max_traj:

        # sample trajectories from environment
        # and safe in replay buffer
        for _ in range(traj_update_count):

            # sample steps
            while True:
                # action from current policy
                a, logp = get_action(s)
                # do step in environment
                s2, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1

                d = False if ep_len == max_ep_len else d

                replay_buffer.store(s.reshape(state_dim), a, r, logp.cpu().numpy())
                s = s2
                # reset environment if necessary
                if d or (ep_len == max_ep_len):
                    s, ep_ret, ep_len = env.reset(), 0, 0
                    replay_buffer.next_traj()
                    break

        performed_trajectories += traj_update_count
        for k in range(update_target_after):

            # perform gradient descent step
            update_q()
            update(prep_data())

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
                           replay_buffer.stored_interactions)
        logger.log_tabular('TestEpLen', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)

        logger.log_tabular('LossQ', with_min_and_max=False)
        logger.log_tabular('LossEta', with_min_and_max=False)
        logger.log_tabular('LossPi', with_min_and_max=False)
        logger.log_tabular('LossLagr', with_min_and_max=False)

        logger.log_tabular('EtaMu', with_min_and_max=False)
        logger.log_tabular('EtaSig', with_min_and_max=False)
        logger.log_tabular('Eta', with_min_and_max=False)

        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        start_time = time.time()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid_q', type=int, default=200)
    parser.add_argument('--hid_pi', type=int, default=100)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default='mpo')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, "./", True)

    torch.set_num_threads(torch.get_num_threads())
    mpo(lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes_q=[args.hid_q] * args.l,
                       hidden_sizes_pi=[args.hid_pi] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs
        )
