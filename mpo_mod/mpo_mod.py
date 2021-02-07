import itertools
from copy import deepcopy
from itertools import chain
import numpy as np
from tqdm import tqdm
import torch
from torch.distributions import Independent, Normal
from mpo_td0.core import MLPActorCritic
from common.tray_dyn_buf import DynamicTrajectoryBuffer


def mpo_runner(
        env,
        writer,
        q_update,
        pi_update,
        min_steps_per_epoch=4000,
        episode_len=200,
        batch_act=20,
        batch_s=768,
        batch_q=1,
        episodes=20,
        epochs=20,
        update_inner=4,
        update_q_after=50,
        update_pi_after=25,
        local_device='cuda:0'):
    run = 0
    iteration = 0

    eta = torch.tensor([1.0], device=local_device, requires_grad=True)
    eta_mean_t = torch.tensor([0.0], device=local_device, requires_grad=True)
    eta_cov_t = torch.tensor([0.0], device=local_device, requires_grad=True)

    ac = MLPActorCritic(env, local_device).to(device=local_device)
    ac_targ = deepcopy(ac).to(device=local_device)

    for p in ac_targ.parameters():
        p.requires_grad = False
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    actor_optimizer = torch.optim.Adam(chain(ac.pi.parameters(), [eta]), lr=lr_pi)
    critic_optimizer = torch.optim.Adam(q_params, lr=lr_q)
    lagrauge_optimizer = torch.optim.Adam(chain([eta_mean_t], [eta_cov_t]), lr=0.01)

    ds = env.observation_space.shape[0]
    da = env.action_space.shape[0]

    replay_buffer = DynamicTrajectoryBuffer(ds,
                                            da,
                                            1,
                                            episode_len,
                                            1,
                                            5000,
                                            local_device)


    performed_steps = 0
    for it in range(iteration, epochs):
        # Find better policy by gradient descent
        while performed_steps < min_steps_per_epoch:
            performed_steps += sample_traj()
        writer.add_scalar('performed_steps', replay_buffer.stored_interactions(), it)

        for r in tqdm(update_steps, desc='updating nets'):

            # update target networks
            if r % update_after == 0:
                for target_param, param in zip(ac_targ.parameters(), ac.parameters()):
                    target_param.data.copy_(param.data)

            # update q function with td0
            q_update(r)
            pi_update(r)


        test_agent(it)
        writer.add_scalar('performed_steps', replay_buffer.stored_interactions(), it)
        writer.flush()
        it += 1
