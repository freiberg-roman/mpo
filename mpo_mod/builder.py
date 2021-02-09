from mpo_mod.loss_fn import UpdateQ_TD0, PolicyUpdateNonParametric
from mpo_mod.helper_fn import SamplerTrajectory, TestAgent, TargetAction
import gym
from mpo_mod.mpo_mod import mpo_runner
from mpo_mod.core_double import MLPActorCritic
from copy import deepcopy
import itertools
import torch
from common.tray_dyn_buf import DynamicTrajectoryBuffer

episode_len = {
    'Pendulum-v0': (200, 200),
    'HalfCheetah-v2': (1000, 1000),
    'Ant-v2': (10, 1000),
}


def mpo_non_parametric_td0(env_name,
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
                           total_steps=40000,
                           min_steps_per_epoch=200,
                           test_after=4000,
                           update_steps=1200,
                           update_after=100
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
    replay_buffer = DynamicTrajectoryBuffer(ds,
                                            da,
                                            min,
                                            max,
                                            min,
                                            20000,
                                            local_device)

    # prepare modules
    q_update = UpdateQ_TD0(
        writer,
        critic_optimizer,
        ac,
        ac_targ,
        replay_buffer,
        batch_size,
        gamma,
        0.0
    )

    pi_update = PolicyUpdateNonParametric(local_device,
                                          writer,
                                          ac,
                                          ac_targ,
                                          actor_optimizer,
                                          eta,
                                          eps_mean,
                                          eps_cov,
                                          eps_dual,
                                          lr_kl,
                                          replay_buffer,
                                          batch_size,
                                          batch_size_act,
                                          ds,
                                          da
                                          )
    actor_step = TargetAction(
        local_device,
        ac_targ,
        ds
    )

    sampler = SamplerTrajectory(
        env,
        local_device,
        writer,
        replay_buffer,
        actor_step,
        max
    )

    test_agent = TestAgent(env, writer, max, actor_step)

    return lambda: mpo_runner(writer,
                              q_update,
                              pi_update,
                              sampler,
                              test_agent,
                              ac,
                              ac_targ,
                              replay_buffer,
                              total_steps,
                              min_steps_per_epoch,
                              test_after,
                              update_steps,
                              update_after,
                              )
