import torch
import numpy as np
from tqdm import tqdm


class Sampler:
    """
    Capsules all information required to perform steps in specified environment.
    """

    def __init__(self, env, device, writer, buffer, actor_step, sample_first, sample_min, max_ep_len, ac_targ):
        """

        @param env: environment compatible with Open AI gym API
        @param device: either 'cuda:0' or 'cpu'
        @param writer: SummaryWriter defined by tensorboard
        @param buffer: replay buffer of any kind
        @param actor_step: encapsulated function that allows to run model
        @param sample_first: amount of steps to sample on first call
        @param sample_min: minimal amount to be sampled per call
        @param max_ep_len: maximal lenght of episode before reset of environment
        @param ac_targ: target actor critic
        """

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

        # internal variables
        self.s = self.env.reset()
        self.ep_len = 0

    def __call__(self):
        """
        Performs and stores steps from the environment.

        @return: returns amount of performed steps in the environment
        """

        performed_steps = 0
        to_perform_steps = self.sample_min
        if self.first_run:
            to_perform_steps = self.sample_first
            self.first_run = False

        while performed_steps < to_perform_steps:
            a, logp = self.actor_step(self.s)
            # do step in environment
            s2, r, d, _ = self.env.step(a.reshape(1, self.da).detach().cpu().numpy())
            self.ep_len += 1
            performed_steps += 1

            self.buffer.store(
                self.s.reshape(self.ds),
                s2.reshape(self.ds),
                a.detach().cpu().numpy(),
                r,
                logp.detach().cpu().numpy(),
                d)

            # update state
            self.s = s2

            # end of trajectory handling
            if d or self.ep_len == self.max_ep_len:
                self.s = self.env.reset()
                self.ep_len = 0
                self.buffer.next_traj()
        return performed_steps


class TargetActionSAC:
    """
    Encapsulates soft actor critic model to be used for sampling
    """

    def __init__(self, device, ac_targ, ds):
        """
        Initializes sampler

        @param device: either 'cuda:0' or 'cpu'
        @param ac_targ: target actor critic
        @param ds: state dimension defined by environment
        """

        self.device = device
        self.actor = ac_targ
        self.ds = ds

    def __call__(self, state, deterministic=False):
        """
        Performs action from model

        @param state: state from environment
        @param deterministic: defines whether to use stochastic or deterministic function
        @return: returns action from model
        """

        act, logp = self.actor.pi.forward(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=self.device).reshape(1, self.ds),
            deterministic=deterministic)
        return act, logp


class TargetActionMPO:
    """
    Encapsulates mpo model to be used for sampling.
    """

    def __init__(self, device, ac_targ, ds):
        """
        Initializes sampler

        @param device: either 'cuda:0' or 'cpu'
        @param ac_targ: target actor critic
        @param ds: state dimension defined by environment
        """

        self.device = device
        self.actor = ac_targ
        self.ds = ds

    def __call__(self, state, deterministic=False):
        """
        Performs action from model

        @param state: state from environment
        @param deterministic: defines whether to use stochastic or deterministic function
        @return: returns action from model
        """

        mean, chol = self.actor.pi.forward(
            torch.as_tensor(state,
                            dtype=torch.float32,
                            device=self.device).reshape(1, self.ds))
        if deterministic:
            return mean, None
        act = self.actor.get_act(mean, chol).squeeze()
        return act, self.actor.get_logp(mean, chol, act)


class TestAgent:
    """
    Encapsulates function to test trained model in separate environment.
    """

    def __init__(self, env, writer, max_ep_len, target_action):
        """
        Initializes testing agent.

        @param env: Open AI gym environment
        @param writer: SummaryWriter from tensorboard for logging
        @param max_ep_len: maximum length of episode while testing
        @param target_action: encapsulated model function to perform actions for evaluation
        """

        self.env = env
        self.writer = writer
        self.max_ep_len = max_ep_len
        self.action = target_action
        self.da = env.action_space.shape[0]

    def __call__(self, run):
        """
        Evaluates model on call with 200 test episodes.

        @param run: run of test in learning process
        """
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
        print('test_ep_ret:', np.array(ep_ret_list).mean(), ' run:', run + 1)
