import mpo_mod.helper_fn
import mpo_mod.loss_fn
import gym
from mpo_mod.mpo_mod import mpo_runner
from mpo_mod.core import MLPActorCritic
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
                                            1,
                                            episode_len,
                                            1,
                                            5000,
                                            local_device)

    # prepare modules

    return lambda : mpo_runner(

    )
