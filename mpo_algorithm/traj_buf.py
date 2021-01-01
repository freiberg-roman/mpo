import numpy as np
import torch


class TrajtoryBuffer:
    def __init__(self, state_dim, action_dim, rollout, traj_size, device):
        self.s_buf = np.zeros((traj_size, rollout, state_dim),
                              dtype=np.float32)
        self.action_buf = np.zeros((traj_size, rollout, action_dim),
                                   dtype=np.float32)
        self.rew_buf = np.zeros((traj_size, rollout, 1))
        self.pi_logp_buf = np.zeros((traj_size, rollout, 1))

        self.ptr_traj, self.ptr_step = 0, 0
        self.max_traj, self.max_rollout = traj_size, rollout
        self.size = 0
        self.device = device

    def store(self, state, action, reward, pi_logp):
        self.s_buf[self.ptr_traj, self.ptr_step, :] = state
        self.action_buf[self.ptr_traj, self.ptr_step, :] = action
        self.action_buf[self.ptr_traj, self.ptr_step] = reward
        self.pi_logp_buf[self.ptr_traj, self.ptr_step] = pi_logp
        self.rew_buf[self.ptr_traj, self.ptr_step] = reward

        self.ptr_step += 1
        if self.ptr_step == self.max_rollout:
            self.ptr_step = 0
            self.ptr_traj = (self.ptr_traj + 1) % self.max_traj
            self.size = min(self.size + 1, self.max_traj)

    def next_traj(self):
        pass

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size * self.max_rollout,
                                 size=batch_size)
        col, row = idxs // self.size, idxs % self.size
        batch = dict(
            state=self.s_buf[row, col],
            action=self.action_buf[row, col],
            reward=self.rew_buf[row, col],
            pi_logp=self.pi_logp_buf[row, col],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                for k, v in batch.items()}

    def sample_traj(self, traj_batch_size=32):
        idxs = np.random.randint(0, self.size, size=traj_batch_size)
        batch = dict(
            state=self.s_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.rew_buf[idxs],
            pi_logp=self.pi_logp_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                for k, v in batch.items()}
