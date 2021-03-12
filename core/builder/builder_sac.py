from core.q_loss.q_loss_fn import UpdateQ_TDE
from core.policy_loss.loss_sac import PolicySACUpdate
from core.helper_fn import Sampler, TestAgent, TargetActionSAC
import gym
from core.main_loop import runner
from core.nets.nets_sac_double_q import MLPActorCriticSAC
from copy import deepcopy
import itertools
import torch
from common.simple_buffer import SimpleBuffer
from core.builder.environment_settings import episode_len


def sac(env_name,
        local_device,
        writer,
        batch_size=100,
        update_steps=50,
        min_steps_per_epoch=50,
        polyak=0.995,
        lr=1e-3,
        entropy=0.2,
        gamma=0.99,
        total_steps=12000,
        test_after=4000,
        update_after=50):
    """
    Soft Actor Critic (SAC) implementation using this framework. Whole algorithm is conceptually
    identical to Open AI's spinning up implementation. This implementation does not claim maximum performance.
    It is simply used to establish a simple baseline to compare MPO against by using a similar architecture.

    @param env_name: string with the environment to train and test on
    @param local_device: either 'cuda:0' or 'cpu'. Note: cpu is still used in Open AI gym
    @param writer: SummaryWriter from tensorboard used for logging. Valid stub can also be used

    @param batch_size: batch size of states sampled per update step
    @param update_steps: amount of updates in each iteration (before next sampling step)
    @param min_steps_per_epoch: minimal amount of steps that will be sampled during one iteration of learning
    @param polyak: parameter for running averages in target updates for q functions and policy
    @param lr: Adam learning rate for both policy and q functions
    @param entropy: also known as alpha temperature parameter which controls the influence of entropy in reward
    @param gamma: discount factor used in TD0
    @param total_steps: minimal amount of steps that will be performed during training phase
    @param test_after: minimal amount of steps that will be performed before evaluating model
    @param update_after: this parameter is not used and only serves as a stub

    @return: runner function to start learning process
    """
    env = gym.make(env_name)
    env_test = gym.make(env_name)
    ac = MLPActorCriticSAC(env.observation_space, env.action_space).to(device=local_device)
    # ac = MLPActorCritic(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    critic_optimizer = torch.optim.Adam(q_params, lr=lr)
    actor_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=lr)

    da = env.action_space.shape[0]
    ds = env.observation_space.shape[0]

    _, max = episode_len[env_name]
    replay_buffer = SimpleBuffer(state_dim=ds,
                                 action_dim=da,
                                 device=local_device)

    # prepare modules
    q_update = UpdateQ_TDE(
        writer=writer,
        critic_optimizer=critic_optimizer,
        ac=ac,
        ac_targ=ac_targ,
        buffer=replay_buffer,
        batch_size=batch_size,
        gamma=gamma,
        entropy=entropy
    )

    pi_update = PolicySACUpdate(
        writer=writer,
        buffer=replay_buffer,
        ac=ac,
        actor_optimizer=actor_optimizer,
        entropy=entropy,
        batch_size=batch_size
    )

    actor_step = TargetActionSAC(
        device=local_device,
        ac_targ=ac,
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
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def target_update():
        pass

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

