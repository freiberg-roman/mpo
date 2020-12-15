from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from mpo_algorithm import core
from utils.logx import EpochLogger
from mpo_algorithm.retrace import Retrace


class ReplayBuffer:
    """
    Simple FIFO replay buffer for MPO
    """

    def __init__(self, s_dim, act_dim, size):
        # state observed
        self.s_buf = np.zeros(core.combined_shape(size, s_dim), dtype=np.float32)
        # action performed for this state
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        # reward for that state and action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # log(π(a|s)) at the time of selection
        self.pi_logp = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, pi_prob, done):
        self.s_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.pi_logp[self.ptr] = pi_prob
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.s_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     pi_logp=self.pi_logp[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def mpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        replay_size=int(1e6), gamma=0.99, max_traj=100000, traj_update_count=500,
        max_ep_len=1000, eps=0.1, eps_mu=0.1, eps_sig=0.0001, lr=0.0005,
        alpha=0.1, batch_size_replay=100, batch_size_policy=100, init_eta=10.0,
        init_eta_mu=10.0, init_eta_sigma=100.0, update_target_after=50,
        num_test_episodes=10, logger_kwargs=dict(), save_freq=1):
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    torch.autograd.set_detect_anomaly(True)

    # create actor-critic module and target networks
    if ac_kwargs is not dict():
        hid_q = ac_kwargs['hidden_sizes_q']
        hid_pi = ac_kwargs['hidden_sizes_pi']
    else:
        hid_q = (200, 200)
        hid_pi = (100, 100)
    ac = actor_critic(env.observation_space, env.action_space,
                      hidden_sizes_q=hid_q, hidden_sizes_pi=hid_pi)
    ac_targ = deepcopy(ac)

    # setting up values where gradient is required
    eta = torch.tensor([init_eta], requires_grad=True)
    eta_sig = torch.tensor([init_eta_sigma], requires_grad=True)
    eta_mu = torch.tensor([init_eta_mu], requires_grad=True)

    # no update for target network with respect to optimizers (copied after k
    # gradient descent steps)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # replay buffer
    replay_buffer = ReplayBuffer(s_dim=state_dim, act_dim=action_dim, size=replay_size)

    # counting variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    # loss for q function will be computed with retrace implementation provided
    # by Denis Bless
    # the implementation provides multiple batches of trajectories
    # only one will be used here
    # result: L(φ) for computing ∂φ L'(φ) (Q function gradient)
    def compute_loss_q(data):
        q = torch.reshape(data['q'], (1, batch_size_replay))
        # 𝔼_π Q(s_t,.) is estimated with values samples by policy i.e
        # 1/batch_size_policy * sum(Q(s_t,a_pi)) over a_pi in batch
        # where Q is the current Q function
        # note: the states are the same as used for q_values
        expected_q = torch.reshape(data['expected_q'], (1, batch_size_replay))
        # values used from delayed network (target network)
        q_target = torch.reshape(data['tar_q'], (1,batch_size_replay))
        rew = torch.reshape(data['rew'], (1, batch_size_replay))
        # π(a|s_t) values from current policy
        # note: the states are the same as used for q_values
        target_pi_prob = torch.reshape(data['pi'], (1, batch_size_replay))
        # behaviour policy values b(a|s_t) are emulated by using old values
        # in replay buffer
        behaviour_policy_prob = torch.reshape(data['b'], (1, batch_size_replay))
        retrace = Retrace()

        q_info = dict(QVals=q.detach().numpy())
        return retrace(q, expected_q, q_target, rew, target_pi_prob,
                       behaviour_policy_prob, gamma), q_info

    # computes function g(eta)
    # q_values: tensor of shape (batch_size_replay, batch_size_policy)
    def compute_eta(q_values):
        return eps * eta + eta * torch.mean(torch.mean(
            torch.exp(q_values / eta), 1  # first mean on batch_size_policy
        )), dict(Eta=eta.detach().numpy())  # then mean on batch_size_replay

    # assume that all target_sigma_values and current_sigma_values are of shape
    # [B, [n,]] where B is the batch size for integral estimation and [n,] is the
    # diagonal covariance matrices
    def compute_eta_sig(data):
        target_sigma = data['tar_sig']
        current_sigma = data['cur_sig']
        torch.squeeze(target_sigma, -1)
        torch.squeeze(current_sigma, -1)
        assert target_sigma.shape == current_sigma.shape

        n = target_sigma.shape[1]  # double check this one
        combined_trace = torch.sum(1 / current_sigma * target_sigma,
                                   dim=1)
        target_det = torch.prod(target_sigma, dim=1)
        current_det = torch.prod(current_sigma, dim=1)
        log_det = torch.log(current_det / target_det)
        c_sig = 0.5 * torch.mean(combined_trace - n + log_det)
        return eta_sig * (eps_sig - c_sig), dict(EtaSig=eta_sig.detach().numpy())

    # same idea as in previous function
    def compute_eta_mu(data):
        current_sigma = data['cur_sig']
        current_mu = data['cur_mu']
        target_mu = data['tar_mu']
        current_sigma = torch.squeeze(current_sigma, dim=-1)
        current_mu = torch.squeeze(current_mu, dim=-1)
        target_mu = torch.squeeze(target_mu, dim=-1)
        assert current_mu.shape == target_mu.shape == current_sigma.shape

        dif = target_mu - current_mu
        c_mu = 0.5 * torch.mean(torch.sum(dif ** 2 * current_sigma, dim=-1))
        return eta_mu * (eps_mu - c_mu), dict(EtaMu=eta_mu.detach().numpy())

    # target_q: tensor of shape (batch_size_replay, batch_size_policy)
    # cur_logp: tensor of shape (batch_size_replay, batch_size_policy)
    def compute_pi_loss(data):
        target_q = data['tar_q']
        cur_logp = data['cur_logp']
        eta.requires_grad = False
        exp_q_eta = torch.exp(target_q / eta)

        res = torch.mean(torch.mean(cur_logp * exp_q_eta, 1))
        eta.requires_grad = True
        return res, dict(PiLoss=res.detach().numpy())

    q_optimizer = Adam(ac.q.parameters(), lr=lr)
    eta_optimizer = Adam([eta], lr=lr)
    eta_lagr_optimizer = Adam([eta_mu, eta_sig], lr=lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)

    logger.setup_pytorch_saver(ac)

    def update(data):
        # gradient step for q function
        q_optimizer.zero_grad()
        q_data = data['q']
        loss_q, q_info = compute_loss_q(q_data)
        loss_q.backward()
        q_optimizer.step()

        # logger
        logger.store(LossQ=loss_q.item(), **q_info)

        # gradient step for eta
        eta_optimizer.zero_grad()
        eta_data = data['eta']
        loss_eta, eta_info = compute_eta(eta_data)
        loss_eta.backward()
        eta_optimizer.step()

        # logger
        logger.store(LossEta=loss_eta, **eta_info)

        # gradient step for eta_mu and eta_sig
        eta_lagr_optimizer.zero_grad()
        eta_lagr_mu_data = data['eta_lagr_mu']
        loss_eta_lagr_mu, mu_info = compute_eta_mu(eta_lagr_mu_data)
        loss_eta_lagr_mu.backward()

        eta_lagr_sig_data = data['eta_lagr_sig']
        loss_eta_lagr_sig, sig_info = compute_eta_sig(eta_lagr_sig_data)
        loss_eta_lagr_sig.backward()
        eta_lagr_optimizer.step()

        # logger
        logger.store(LossEtaMu=loss_eta_lagr_mu, **mu_info)
        logger.store(LossEtaSig=loss_eta_lagr_sig, **sig_info)

        # gradient step for policy
        pi_optimizer.zero_grad()
        pi_data = data['pi']
        loss_pi, pi_info = compute_pi_loss(pi_data)
        loss_pi.backward()
        pi_optimizer.step()

        # logger
        logger.store(LossPi=loss_pi, **pi_info)

    def prep_data(samples):

        # storage for actions from target network for each state in samples
        tar_act_samples = np.zeros((batch_size_replay,
                                    batch_size_policy,
                                    action_dim),
                                   dtype=np.float32)

        # for each state in samples get actions from target network to
        # estimate integrals
        for i, s in enumerate(samples['state']):
            s = torch.reshape(torch.tensor(s).repeat(batch_size_policy),
                              (batch_size_policy, state_dim[0]))
            tar_act, _, _, _ = ac_targ.pi.forward(s)
            tar_act_samples[i, :, :] = tar_act

        # get Q values for samples from current Q function
        # get Q values from target Q function
        tar_act_samples = torch.tensor(tar_act_samples)
        states = torch.tensor(samples['state']).repeat(batch_size_policy, 1)
        states = torch.reshape(states,
                               (batch_size_replay, batch_size_policy, state_dim[0]))
        tar_q_batch = ac_targ.q.forward(states, tar_act_samples)

        # get target and current policy parameters for samples

        for p in ac.parameters():
            p.requires_grad = False

        _, _, tar_mu, tar_sig = ac_targ.pi.forward(samples['state'])
        _, _, cur_mu, cur_sig = ac.pi.forward(samples['state'])

        for p in ac.parameters():
            p.requires_grad = True

        # for estimating policy loss
        cur_logp = ac.pi.get_prob(states, tar_act_samples)

        return dict(
            q=ac.q.forward(samples['state'], samples['action']),
            expected_q=torch.mean(tar_q_batch, dim=-1),
            tar_q=ac_targ.q.forward(samples['state'], samples['action']),
            rew=torch.tensor(samples['reward']),
            pi=torch.squeeze(torch.exp(ac_targ.pi.get_prob(
                samples['state'], samples['action'])), -1),
            b=torch.tensor(torch.exp(samples['pi_logp'])),
            tar_q_batch=tar_q_batch,
            tar_mu=tar_mu,
            tar_sig=tar_sig,
            cur_mu=cur_mu,
            cur_sig=cur_sig,
            cur_logp=cur_logp,
        )

    def get_action(state, deterministic=False):
        action, logp_pi = ac.act(
            torch.as_tensor(state, dtype=torch.float32, ), deterministic)
        return action, logp_pi

    def test_agent():
        for j in range(num_test_episodes):
            s, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                s, r, d, _ = test_env.step(get_action(s, True)[0])
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    s, ep_ret, ep_len = env.reset(), 0, 0

    # main loop
    performed_trajectories = 0
    while performed_trajectories < max_traj:

        # sample trajectories from environment to update replay buffer
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

                replay_buffer.store(s, a, r, logp, d)
                # reset environment if necessary
                if d or (ep_len == max_ep_len):
                    s, ep_ret, ep_len = env.reset(), 0, 0
                    break

        performed_trajectories += traj_update_count

        for k in range(update_target_after):
            # collect data
            data = dict()

            # sample (s, a, r, pi_logp, d) from replay buffer
            samples = replay_buffer.sample_batch(batch_size=batch_size_replay)

            all_data = prep_data(samples)
            data['q'] = all_data
            data['eta'] = all_data['tar_q_batch']
            data['eta_lagr_sig'] = dict(
                tar_sig=all_data['tar_sig'],
                cur_sig=all_data['cur_sig'],
            )
            data['eta_lagr_mu'] = dict(
                cur_sig=all_data['cur_sig'],
                tar_mu=all_data['tar_mu'],
                cur_mu=all_data['cur_mu'],
            )
            data['pi'] = dict(
                tar_q=all_data['tar_q_batch'],
                cur_logp=all_data['cur_logp'],
            )

            # perform gradient descent step
            update(data)

        # update target parameters
        # use ugly hack until better option is found
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(0)
                p_targ.data.add_(p)

        # after each epoch evaluate performance of agent
        test_agent()

        # Log all relevant information
        logger.log_tabular('Performed Trajectories', performed_trajectories)
        logger.log_tabular('TestEpRet', with_min_and_max=True)

        logger.log_tabular('LossQ', with_min_and_max=True)
        logger.log_tabular('LossEta', with_min_and_max=True)
        logger.log_tabular('LossEtaMu', with_min_and_max=True)
        logger.log_tabular('LossEtaSig', with_min_and_max=True)
        logger.log_tabular('LossPi', with_min_and_max=True)

        logger.log_tabular('Eta', with_min_and_max=True)
        logger.log_tabular('EtaMu', with_min_and_max=True)
        logger.log_tabular('EtaSig', with_min_and_max=True)

        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid_q', type=int, default=200)
    parser.add_argument('--hid_pi', type=int, default=100)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--max_traj', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='mpo')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())
    mpo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes_q=[args.hid_q] * args.l,
                       hidden_sizes_pi=[args.hid_pi] * args.l),
        gamma=args.gamma, seed=args.seed, max_traj=args.max_traj,
        logger_kwargs=logger_kwargs)
