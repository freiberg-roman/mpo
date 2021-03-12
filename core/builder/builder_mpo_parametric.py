from core.q_loss.q_loss_fn import UpdateQ_TD, UpdateQRetrace
from core.policy_loss.loss_mpo import PolicyUpdateParametric
from core.helper_fn import Sampler, TestAgent, TargetActionMPO
import gym
from core.main_loop import runner
from core.nets.nets_mpo_double_q import MLPActorCritic
from core.nets.nets_mpo_single_q import MLPActorCriticSingle
from copy import deepcopy
import itertools
import torch
from common.tray_dyn_buf import DynamicTrajectoryBuffer
from common.simple_buffer import SimpleBuffer
from core.builder.environment_settings import episode_len


def mpo_parametric_td0(env_name,
                       local_device,
                       writer,
                       lr_pi=5e-4,
                       lr_q=5e-4,
                       lr_kl=0.01,
                       eps_mean=0.05,
                       eps_cov=0.00001,
                       batch_size=128,
                       batch_size_act=20,
                       gamma=0.99,
                       total_steps=12000,
                       min_steps_per_epoch=50,
                       test_after=4000,
                       update_steps=50,
                       update_after=50,
                       polyak=0.995,
                       ):
    """
    Builds a runnable version of the MPO parametric version using TD0 algorithm
    for updating Q values

    @param env_name: string with the environment to train and test on
    @param local_device: either 'cuda:0' or 'cpu'. Note: cpu is still used in Open AI gym
    @param writer: SummaryWriter from tensorboard used for logging. Valid stub can also be used

    @param lr_pi: Adam learning rate for policy
    @param lr_q: Adam learning rate for q values
    @param lr_kl: Adam learning rate for lagrange constrains
    @param eps_mean: epsilon KL constrain for mean
    @param eps_cov: epsilon KL constrain for covariance
    @param batch_size: batch size of states sampled per update step
    @param batch_size_act: batch size of actions sampled for each state to estimate expectations
    @param gamma: discount factor used in TD0
    @param total_steps: minimal amount of steps that will be performed during training phase
    @param min_steps_per_epoch: minimal amount of steps that will be sampled during one iteration of learning
    @param test_after: minimal amount of steps that will be performed before evaluating model
    @param update_steps: amount of updates in each iteration (before next sampling step)
    @param update_after: amount of updates after which the target model will be copied over (only for policy)
    @param polyak: parameter for running averages in target updates for q functions

    @return: runner function to start learning process
    """
    env = gym.make(env_name)
    env_test = gym.make(env_name)
    ac = MLPActorCritic(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_param = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    critic_optimizer = torch.optim.Adam(q_param, lr=lr_q)
    actor_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=lr_pi)

    # set up replay buffer with min and max trajectory length
    da = env.action_space.shape[0]
    ds = env.observation_space.shape[0]

    min, max = episode_len[env_name]
    replay_buffer = SimpleBuffer(state_dim=ds,
                                 action_dim=da,
                                 device=local_device)

    # prepare modules
    q_update = UpdateQ_TD(
        writer=writer,
        critic_optimizer=critic_optimizer,
        ac=ac,
        ac_targ=ac_targ,
        buffer=replay_buffer,
        batch_size=batch_size,
        gamma=gamma,
        entropy=0.0
    )

    pi_update = PolicyUpdateParametric(
        device=local_device,
        writer=writer,
        ac=ac,
        ac_targ=ac_targ,
        actor_eta_optimizer=actor_optimizer,
        eps_mean=eps_mean,
        eps_cov=eps_cov,
        lr_kl=lr_kl,
        buffer=replay_buffer,
        batch_size=batch_size,
        batch_size_act=batch_size_act,
        ds=ds,
        da=da
    )
    actor_step = TargetActionMPO(
        device=local_device,
        ac_targ=ac_targ,
        ds=ds
    )

    sampler = Sampler(
        env=env,
        device=local_device,
        writer=writer,
        buffer=replay_buffer,
        actor_step=actor_step,
        sample_first=1000,
        sample_min=min_steps_per_epoch,
        max_ep_len=max,
        ac_targ=ac_targ
    )

    test_agent = TestAgent(env_test, writer, max, actor_step)

    def one_step_update():
        for p, p_targ in zip(ac.q1.parameters(), ac_targ.q1.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

        for p, p_targ in zip(ac.q2.parameters(), ac_targ.q2.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

    def target_update():
        for target_param, param in zip(ac_targ.pi.parameters(), ac.pi.parameters()):
            target_param.data.copy_(param.data)

    return lambda: runner(writer=writer,
                          q_update=q_update,
                          pi_update=pi_update,
                          one_step_update=one_step_update,
                          target_update=target_update,
                          sampler=sampler,
                          test_agent=test_agent,
                          buffer=replay_buffer,
                          total_steps=total_steps,
                          min_steps_per_iteration=min_steps_per_epoch,
                          test_after=test_after,
                          update_steps=update_steps,
                          update_after=update_after,
                          )


def mpo_parametric_retrace(env_name,
                           local_device,
                           writer,
                           lr_pi=5e-4,
                           lr_q=5e-4,
                           lr_kl=0.01,
                           eps_mean=0.05,
                           eps_cov=0.00001,
                           batch_size=128,
                           batch_size_act=20,
                           gamma=0.99,
                           total_steps=300000,
                           min_steps_per_epoch=1000,
                           test_after=4000,
                           update_steps=50,
                           update_after=50,
                           rollout_len=5,
                           polyak=0.995,
                           ):
    """
    Builds a runnable version of the MPO parametric version using TD0 algorithm
    for updating Q values

    @param env_name: string with the environment to train and test on
    @param local_device: either 'cuda:0' or 'cpu'. Note: cpu is still used in Open AI gym
    @param writer: SummaryWriter from tensorboard used for logging. Valid stub can also be used

    @param lr_pi: Adam learning rate for policy
    @param lr_q: Adam learning rate for q values
    @param lr_kl: Adam learning rate for lagrange constrains
    @param eps_mean: epsilon KL constrain for mean
    @param eps_cov: epsilon KL constrain for covariance
    @param batch_size: batch size of states sampled per update step
    @param batch_size_act: batch size of actions sampled for each state to estimate expectations
    @param gamma: discount factor used in TD0
    @param total_steps: minimal amount of steps that will be performed during training phase
    @param min_steps_per_epoch: minimal amount of steps that will be sampled during one iteration of learning
    @param test_after: minimal amount of steps that will be performed before evaluating model
    @param update_steps: amount of updates in each iteration (before next sampling step)
    @param update_after: amount of updates after which the target model will be copied over (only for policy)
    @param rollout_len: length of sample trajectories used in Retrace algorithm
    @param polyak: parameter for running averages in target updates for q functions

    @return: runner function to start learning process
    """
    env = gym.make(env_name)
    env_test = gym.make(env_name)
    ac = MLPActorCriticSingle(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False

    critic_optimizer = torch.optim.Adam(ac.q.parameters(), lr=lr_q)
    actor_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=lr_pi)

    # set up replay buffer with min and max trajectory length
    da = env.action_space.shape[0]
    ds = env.observation_space.shape[0]

    min, max = episode_len[env_name]
    replay_buffer = DynamicTrajectoryBuffer(
        ds, da, min, max, rollout_len, 2000, local_device
    )

    # prepare modules
    q_update = UpdateQRetrace(
        writer=writer,
        critic_optimizer=critic_optimizer,
        ac=ac,
        ac_targ=ac_targ,
        buffer=replay_buffer,
        batch_size=batch_size,
        gamma=gamma,
        device=local_device
    )

    pi_update = PolicyUpdateParametric(
        device=local_device,
        writer=writer,
        ac=ac,
        ac_targ=ac_targ,
        actor_eta_optimizer=actor_optimizer,
        eps_mean=eps_mean,
        eps_cov=eps_cov,
        lr_kl=lr_kl,
        buffer=replay_buffer,
        batch_size=batch_size,
        batch_size_act=batch_size_act,
        ds=ds,
        da=da
    )
    actor_step = TargetActionMPO(
        device=local_device,
        ac_targ=ac_targ,
        ds=ds
    )

    sampler = Sampler(
        env=env,
        device=local_device,
        writer=writer,
        buffer=replay_buffer,
        actor_step=actor_step,
        sample_first=1000,
        sample_min=min_steps_per_epoch,
        max_ep_len=max,
        ac_targ=ac_targ
    )

    test_agent = TestAgent(env_test, writer, max, actor_step)

    def one_step_update():
        for p, p_targ in zip(ac.q.parameters(), ac_targ.q.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)

    def target_update():
        for target_param, param in zip(ac_targ.pi.parameters(), ac.pi.parameters()):
            target_param.data.copy_(param.data)

    return lambda: runner(writer=writer,
                          q_update=q_update,
                          pi_update=pi_update,
                          one_step_update=one_step_update,
                          target_update=target_update,
                          sampler=sampler,
                          test_agent=test_agent,
                          buffer=replay_buffer,
                          total_steps=total_steps,
                          min_steps_per_iteration=min_steps_per_epoch,
                          test_after=test_after,
                          update_steps=update_steps,
                          update_after=update_after,
                          )

