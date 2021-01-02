import numpy as np
import torch


class DynamicTrajectoryBuffer:
    def __init__(self,
                 state_dim,
                 action_dim,
                 min_rollout,
                 max_rollout,
                 traj_size,
                 device):
        self.s_buf = np.zeros((traj_size, max_rollout, state_dim),
                              dtype=np.float32)
        self.action_buf = np.zeros((traj_size, max_rollout, action_dim),
                                   dtype=np.float32)
        self.rew_buf = np.zeros((traj_size, max_rollout, 1),
                                dtype=np.float32)
        self.pi_logp_buf = np.zeros((traj_size, max_rollout, 1),
                                    dtype=np.float32)

        self.len_used = np.zeros(traj_size, dtype=np.int32)

        self.device = device
        self.ptr_traj, self.ptr_step = 0, 0
        self.max_traj = traj_size
        self.max_rollout = max_rollout
        self.min_rollout = min_rollout
        self.size_traj = 0
        self.stored_interactions = 0

    def store(self, state, action, reward, pi_logp):
        assert self.ptr_step < self.max_rollout

        self.s_buf[self.ptr_traj, self.ptr_step, :] = state
        self.action_buf[self.ptr_traj, self.ptr_step, :] = action
        self.rew_buf[self.ptr_traj, self.ptr_step, :] = reward
        self.pi_logp_buf[self.ptr_traj, self.ptr_step, :] = pi_logp

        self.ptr_step += 1

    def next_traj(self):
        # will commit the current trajectory
        # only then batch samples from this trajectory are possible
        assert self.ptr_step >= self.min_rollout
        self.ptr_traj = (self.ptr_traj + 1) % self.max_traj
        self.len_used[self.ptr_traj] = self.ptr_step  # length of this trajectory
        self.stored_interactions += self.ptr_step
        self.ptr_step = 0
        self.size_traj = min(self.size_traj + 1, self.max_traj)

    def sample_batch(self, batch_size=32):
        size_all = np.sum(self.len_used)
        idxs = np.random.randint(0, size_all, size=batch_size)

        cum_len_idxs = np.cumsum(self.len_used) - 1
        rows = np.searchsorted(cum_len_idxs[1:], idxs)
        cols = idxs - (cum_len_idxs[rows] + 1)

        batch = dict(
            state=self.s_buf[rows, cols],
            action=self.action_buf[rows, cols],
            reward=self.rew_buf[rows, cols],
            pi_logp=self.pi_logp_buf[rows, cols],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32,
                                   device=self.device) for k, v in batch.items()}

    def sample_traj(self, traj_batch_size=4):
        idxs = np.random.randint(0, self.size_traj, size=traj_batch_size)
        min_length = np.min(self.len_used[1:self.size_traj + 1])
        assert min_length >= self.min_rollout

        batch = dict(
            state=self.s_buf[idxs, 0:min_length],
            action=self.action_buf[idxs, 0:min_length],
            reward=self.rew_buf[idxs, 0:min_length],
            pi_logp=self.pi_logp_buf[idxs, 0:min_length],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                for k, v in batch.items()}
