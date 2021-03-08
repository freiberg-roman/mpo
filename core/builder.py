from core.q_loss_fn import UpdateQ_TD, UpdateQRetrace
from core.loss_fn import PolicyUpdateNonParametric, PolicySACUpdate
from core.helper_fn import Sampler, TestAgent, TargetAction
import gym
from core.main_loop import runner
from core.nets.nets_mpo_double_q import MLPActorCritic
from core.nets.nets_mpo_single_q import MLPActorCriticSingle
from core.nets.nets_sac_double_q import MLPActorCriticSAC
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
                           total_steps=300000,
                           min_steps_per_epoch=50,
                           test_after=4000,
                           update_steps=50,
                           update_after=50,
                           polyak=0.995
                           ):
    env = gym.make(env_name)
    env_test = gym.make(env_name)
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

    _, max = episode_len[env_name]
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

    pi_update = PolicyUpdateNonParametric(
        device=local_device,
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
                               polyak=0.995,
                               ):
    env = gym.make(env_name)
    env_test = gym.make(env_name)
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

    pi_update = PolicyUpdateNonParametric(
        device=local_device,
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


def mpo_parametric_td0():
    pass


def mpo_parametric_retrace():
    pass


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
    env = gym.make(env_name)
    env_test = gym.make(env_name)
    # ac = MLPActorCriticSAC(env.observation_space, env.action_space).to(device=local_device)
    ac = MLPActorCritic(env, local_device).to(device=local_device)
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
    q_update = UpdateQ_TD(
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
