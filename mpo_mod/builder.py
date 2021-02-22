from mpo_mod.loss_fn import UpdateQ_TD0, PolicyUpdateNonParametric, UpdateQRetrace
from mpo_mod.helper_fn import SamplerTrajectory, Sampler, TestAgent, TargetAction
import gym
from mpo_mod.mpo_mod import mpo_runner
from mpo_mod.core import MLPActorCritic
from mpo_mod.core_single import MLPActorCriticSingle
from copy import deepcopy
import itertools
import torch
from common.tray_dyn_buf import DynamicTrajectoryBuffer
from common.simple_buffer import SimpleBuffer

episode_len = {
    'Pendulum-v0': (200, 200),
    'HalfCheetah-v2': (1000, 1000),
    'Ant-v2': (10, 1000),
}


def mpo_non_parametric_td0_sac_update(env_name,
                                      local_device,
                                      writer,
                                      lr_pi=5e-4,
                                      lr_q=5e-4,
                                      lr_kl=0.01,
                                      eps_dual=0.1,
                                      eps_mean=0.1,
                                      eps_cov=0.0001,
                                      batch_size=768,
                                      batch_size_act=20,
                                      gamma=0.99,
                                      total_steps=300000,
                                      min_steps_per_epoch=50,
                                      test_after=4000,
                                      update_steps=50,
                                      update_after=50,
                                      ):
    env = gym.make(env_name)
    ac = MLPActorCritic(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    eta = torch.tensor([1.0], device=local_device, requires_grad=True)
    critic_optimizer = torch.optim.Adam(q_params, lr=lr_q)
    actor_optimizer = torch.optim.Adam(
        itertools.chain(ac.pi.parameters(), [eta]), lr=lr_pi)

    # set up replay buffer with min and max trajectory length
    da = env.action_space.shape[0]
    ds = env.observation_space.shape[0]

    min, max = episode_len[env_name]
    replay_buffer = SimpleBuffer(state_dim=ds,
                                 action_dim=da,
                                 device=local_device)

    # prepare modules
    q_update = UpdateQ_TD0(
        writer=writer,
        critic_optimizer=critic_optimizer,
        ac=ac,
        ac_targ=ac_targ,
        buffer=replay_buffer,
        batch_size=batch_size,
        gamma=gamma,
        entropy=0.0
    )

    pi_update = PolicyUpdateNonParametric(device=local_device,
                                          writer=writer,
                                          ac=ac,
                                          ac_targ=ac_targ,
                                          actor_eta_optimizer=actor_optimizer,
                                          eta=eta,
                                          eps_mean=eps_mean,
                                          eps_cov=eps_cov,
                                          eps_dual=eps_dual,
                                          lr_kl=lr_kl,
                                          buffer=replay_buffer,
                                          batch_size=batch_size,
                                          batch_size_act=batch_size_act,
                                          ds=ds,
                                          da=da
                                          )
    actor_step = TargetAction(
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
        sample_min=50,
        max_ep_len=max
    )

    test_agent = TestAgent(env, writer, max, actor_step)

    return lambda: mpo_runner(writer=writer,
                              q_update=q_update,
                              pi_update=pi_update,
                              sampler=sampler,
                              test_agent=test_agent,
                              ac=ac,
                              ac_targ=ac_targ,
                              buffer=replay_buffer,
                              total_steps=total_steps,
                              min_steps_per_iteration=min_steps_per_epoch,
                              test_after=test_after,
                              update_steps=update_steps,
                              update_after=update_after,
                              )


def mpo_non_parametric_retrace(env_name,
                               local_device,
                               writer,
                               lr_pi=5e-4,
                               lr_q=5e-4,
                               lr_kl=0.01,
                               eps_dual=0.1,
                               eps_mean=0.1,
                               eps_cov=0.0001,
                               batch_size=768,
                               batch_size_act=20,
                               gamma=0.99,
                               total_steps=300000,
                               min_steps_per_epoch=1000,
                               test_after=4000,
                               update_steps=50,
                               update_after=50,
                               rollout_len=5,
                               ):
    env = gym.make(env_name)
    ac = MLPActorCriticSingle(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False

    eta = torch.tensor([1.0], device=local_device, requires_grad=True)
    critic_optimizer = torch.optim.Adam(ac.q.parameters(), lr=lr_q)
    actor_optimizer = torch.optim.Adam(
        itertools.chain(ac.pi.parameters(), [eta]), lr=lr_pi)

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

    pi_update = PolicyUpdateNonParametric(device=local_device,
                                          writer=writer,
                                          ac=ac,
                                          ac_targ=ac_targ,
                                          actor_eta_optimizer=actor_optimizer,
                                          eta=eta,
                                          eps_mean=eps_mean,
                                          eps_cov=eps_cov,
                                          eps_dual=eps_dual,
                                          lr_kl=lr_kl,
                                          buffer=replay_buffer,
                                          batch_size=batch_size,
                                          batch_size_act=batch_size_act,
                                          ds=ds,
                                          da=da
                                          )
    actor_step = TargetAction(
        device=local_device,
        ac_targ=ac_targ,
        ds=ds
    )

    sampler = SamplerTrajectory(
        env=env,
        device=local_device,
        writer=writer,
        buffer=replay_buffer,
        actor_step=actor_step,
        max_ep_len=max
    )

    test_agent = TestAgent(env, writer, max, actor_step)

    return lambda: mpo_runner(writer=writer,
                              q_update=q_update,
                              pi_update=pi_update,
                              sampler=sampler,
                              test_agent=test_agent,
                              ac=ac,
                              ac_targ=ac_targ,
                              buffer=replay_buffer,
                              total_steps=total_steps,
                              min_steps_per_iteration=min_steps_per_epoch,
                              test_after=test_after,
                              update_steps=update_steps,
                              update_after=update_after,
                              )
