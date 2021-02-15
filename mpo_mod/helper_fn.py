import torch
import numpy as np
from tqdm import tqdm
from mpo_mod.core import GaussianMLPActor


class SamplerTrajectory:
    def __init__(self, env, device, writer, buffer, actor_step, max_ep_len):
        self.env = env
        self.writer = writer
        self.device = device
        self.buffer = buffer
        self.actor_step = actor_step
        self.da = env.action_space.shape[0]
        self.ds = env.observation_space.shape[0]
        self.max_ep_len = max_ep_len

    def __call__(self):
        s, _, ep_len = self.env.reset(), 0, 0
        while True:
            a, logp = self.actor_step(s)
            # do step in environment
            s2, r, d, _ = self.env.step(a.reshape(1, self.da).cpu().numpy())
            ep_len += 1

            self.buffer.store(
                s.reshape(self.ds),
                s2.reshape(self.ds),
                a.cpu().numpy(),
                r,
                logp.cpu().numpy(),
                d)

            # update state
            s = s2

            # end of trajectory handling
            if ep_len == self.max_ep_len or d:
                self.buffer.next_traj()
                return ep_len


class Sampler:
    def __init__(self, env, device, writer, buffer, actor_step, sample_first, sample_min, max_ep_len):
        self.env = env
        self.writer = writer
        self.device = device
        self.buffer = buffer
        self.actor_step = actor_step
        self.da = env.action_space.shape[0]
        self.ds = env.observation_space.shape[0]
        self.sample_first = sample_first
        self.sample_min = sample_min
        self.first_run = True
        self.max_ep_len = max_ep_len

    def __call__(self):

        performed_steps = 0
        to_perform_steps = self.sample_min
        if self.first_run:
            to_perform_steps = self.sample_first
            self.first_run = False

        s, _, ep_len = self.env.reset(), 0, 0
        while performed_steps < to_perform_steps:
            a, logp = self.actor_step(s)
            # do step in environment
            s2, r, d, _ = self.env.step(a.reshape(1, self.da).cpu().numpy())
            ep_len += 1
            performed_steps += 1

            self.buffer.store(
                s.reshape(self.ds),
                s2.reshape(self.ds),
                a.cpu().numpy(),
                r,
                logp.cpu().numpy(),
                d)

            # update state
            s = s2

            # end of trajectory handling
            if d or ep_len == self.max_ep_len:
                s = self.env.reset()
        return performed_steps

class TargetAction:
    def __init__(self, device, ac_targ, ds):
        self.device = device
        self.actor = ac_targ.pi
        self.ds = ds

    def __call__(self, state, deterministic=False):
        mean, chol = self.actor.forward(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=self.device).reshape(1, self.ds))
        if deterministic:
            return mean, None
        act = GaussianMLPActor.get_act(mean, chol).squeeze()
        return act, GaussianMLPActor.get_logp(mean, chol, act)


class TestAgent:
    def __init__(self, env, writer, max_ep_len, target_action):
        self.env = env
        self.writer = writer
        self.max_ep_len = max_ep_len
        self.action = target_action
        self.da = env.action_space.shape[0]

    def __call__(self, run):
        ep_ret_list = list()
        for _ in tqdm(range(200), desc="testing model"):
            s, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                with torch.no_grad():
                    s, r, d, _ = self.env.step(
                        self.action(
                            s,
                            deterministic=True)[0].reshape(1, self.da).cpu().numpy())
                ep_ret += r
                ep_len += 1
            ep_ret_list.append(ep_ret)
        self.writer.add_scalar('test_ep_ret', np.array(ep_ret_list).mean(), run)
        print('test_ep_ret:', np.array(ep_ret_list).mean(), ' ', run)
