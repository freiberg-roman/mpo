from copy import deepcopy
import numpy as np
import itertools
import torch
from torch.optim import Adam
import time
from mpo_sac import tanh_core
from torch.utils.tensorboard import SummaryWriter
from mpo_sac.tray_dyn_buf import DynamicTrajectoryBuffer
from tqdm import tqdm
from mpo_sac.retrace import Retrace
from torch.distributions.normal import Normal

local_device = "cpu"


def mpo_sac(env_fn,
            actor_critic=tanh_core.MLPActorCritic,
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
            alpha=0.2,
            batch_t=2,  # sampled trajectories per learning step
            batch_act=20,  # additional samples for integral estimation
            len_rollout=200,
            init_eta=0.5,
            init_eta_mean=1.0,
            init_eta_cov=1.0,
            learning_steps=1000,
            update_targ_nets_after=200,
            num_test_episodes=50,
            polyak=0.995,
            reward_scaling=lambda r: r):
    writer = SummaryWriter(comment='MPO_RETRACE_ENT_LOSS_polyak_pi_multiple')

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
    eta = torch.tensor([init_eta], requires_grad=True)
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
    var_counts = tuple(tanh_core.count_vars(module) for module in [ac.pi, ac.q])
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
                         behaviour_policy_probs=torch.transpose(samples['pi_logp'], 0, 1).squeeze(-1)
                         )
        writer.add_scalar('q_loss', loss_q.item(), run)
        writer.add_scalar('q', targ_q.detach().numpy().mean(), run)
        writer.add_scalar('q_min', targ_q.detach().numpy().min(), run)
        writer.add_scalar('q_max', targ_q.detach().numpy().max(), run)

        return loss_q

    def loss_pi_sac(samples, run):

        cur_mean, cur_std = ac.pi.forward(samples['state'])
        cur_act, cur_act_logp = ac.pi.get_act(cur_mean, cur_std)
        cur_q = ac.q.forward(samples['state'], cur_act)

        loss_pi = (alpha * cur_act_logp - cur_q).mean()

        writer.add_scalar('pi_loss', loss_pi.item(), run)
        writer.add_scalar('pi_logp', cur_act_logp.detach().numpy().mean(), run)
        return loss_pi

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
                sample_traj(perform_traj=1)

            if j % update_targ_nets_after == 0:
                for p, p_targ in zip(ac.q.parameters(), ac_targ.q.parameters()):
                    p_targ.data.copy_(p.data)

            rows, cols = replay_buffer.sample_idxs(batch_size=batch_t)
            samples = replay_buffer.sample_trajectories(rows, cols)

            # update q
            for _ in range(10):
                opti_q.zero_grad()
                loss = loss_q_retrace(samples, run=i * learning_steps + j)
                loss.backward()
                opti_q.step()

            # update pi
            for _ in range(10):
                opti_pi.zero_grad()
                loss = loss_pi_sac(samples, run=i * learning_steps + j)
                loss.backward()
                opti_pi.step()

                for p, p_targ in zip(ac.pi.parameters(), ac_targ.pi.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

        test_agent(i)
        writer.add_scalar('time_per_epoch', time.time() - start_time, i)
        start_time = time.time()
        writer.flush()
