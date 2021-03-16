import torch
import numpy as np


class SimpleBuffer:
    """
    A simple FIFO experience replay buffer
    """

    def __init__(self, state_dim, action_dim, device, size=2000000):
        """
        Initialize replay buffer. SimpleBuffer simply uses numpy arrays to store data from gym environments.

        @param state_dim: state dimension defined by the environment
        @param action_dim: action dimension defined by actor from the environment
        @param device: either 'cuda:0' or 'cpu'
        @param size: amount of new steps that will be stored
        """

        self.last = None
        self.s_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.s_next_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.total_steps = 0
        self.device = device

    def store(self, s, s_next, act, rew, log_p, done):
        """
        Stores a complete step with actor information. For compatibility reasons log_p will be offered as a storage
        option but disregarded since no algorithm using SimpleBuffer requires it.

        @param s: current state from the environment
        @param s_next: next state from the environment
        @param act:  action performed by the actor in the environment
        @param rew: reward from performing (s, act) pair
        @param log_p: logarithmic probability density form actor performing this step
        @param done: 1 or 0 depending if this step is the final step of the trajectory
        """

        self.s_buf[self.ptr] = s
        self.s_next_buf[self.ptr] = s_next
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.total_steps += 1

    def sample_batch(self, batch_size=768):
        """
        Function returns a random batch from stored interactions.

        @param batch_size: size of the batch
        @return: random batch of size batch_size
        """

        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.s_buf[idxs],
                     state_next=self.s_next_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        self.last = {k: torch.as_tensor(v,
                                        dtype=torch.float32,
                                        device=self.device) for k, v in batch.items()}
        return self.last

    def stored_interactions(self):
        """
        Returns the number of stored steps in this buffer.

        @return: amount of stored steps
        """

        return self.total_steps

    def next_traj(self):
        """
        Function does nothing and is only provided for compatibility reasons.
        """

        pass

    def last_batch(self):
        return self.last
