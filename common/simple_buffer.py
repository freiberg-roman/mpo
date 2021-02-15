import torch
import numpy as np


class SimpleBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, state_dim, action_dim, device, size=2000000):
        self.s_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.s_next_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.total_steps = 0
        self.device = device

    def store(self, s, s_next, act, rew, log_p, done):
        self.s_buf[self.ptr] = s
        self.s_next_buf[self.ptr] = s_next
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.total_steps += 1

    def sample_batch(self, batch_size=768):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.s_buf[idxs],
                     state_next=self.s_next_buf[idxs],
                     action=self.act_buf[idxs],
                     reward=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v,
                                   dtype=torch.float32,
                                   device=self.device) for k, v in batch.items()}

    def stored_interactions(self):
        return self.total_steps
