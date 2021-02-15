import numpy as np
import torch


class DynamicTrajectoryBuffer:
    def __init__(self,
                 state_dim,
                 action_dim,
                 min_rollout,
                 max_rollout,
                 traj_rollout,
                 traj_size,
                 device):
        self.s_buf = np.zeros((traj_size, max_rollout, state_dim),
                              dtype=np.float32)
        self.s_next_buf = np.zeros((traj_size, max_rollout, state_dim),
                                   dtype=np.float32)
        self.action_buf = np.zeros((traj_size, max_rollout, action_dim),
                                   dtype=np.float32)
        self.rew_buf = np.zeros((traj_size, max_rollout, 1),
                                dtype=np.float32)
        self.pi_logp_buf = np.zeros((traj_size, max_rollout, 1),
                                    dtype=np.float32)
        self.done_buf = np.zeros((traj_size, max_rollout, 1),
                                 dtype=np.float32)

        self.len_used = np.zeros(traj_size, dtype=np.int32)

        self.device = device
        self.ptr_traj, self.ptr_step = 0, 0
        self.max_traj = traj_size
        self.max_rollout = max_rollout
        self.min_rollout = min_rollout

        assert min_rollout >= traj_rollout
        self.traj_rollout = traj_rollout
        self.size_traj = 0
        self.interactions = 0

    def store(self, state, state_next, action, reward, pi_logp, done):
        assert self.ptr_step < self.max_rollout

        self.s_buf[self.ptr_traj, self.ptr_step, :] = state
        self.s_next_buf[self.ptr_traj, self.ptr_step, :] = state_next
        self.action_buf[self.ptr_traj, self.ptr_step, :] = action
        self.rew_buf[self.ptr_traj, self.ptr_step, :] = reward
        self.pi_logp_buf[self.ptr_traj, self.ptr_step, :] = pi_logp
        self.done_buf[self.ptr_traj, self.ptr_step, :] = done

        self.ptr_step += 1
        self.interactions += 1

    def next_traj(self):
        # will commit the current trajectory
        # only then batch samples from this trajectory are possible
        assert self.ptr_step >= self.min_rollout
        self.len_used[self.ptr_traj] = self.ptr_step  # length of this trajectory
        self.ptr_traj = (self.ptr_traj + 1) % self.max_traj
        self.ptr_step = 0
        self.size_traj = min(self.size_traj + 1, self.max_traj)

    def sample_idxs(self, batch_size=768):
        # usable length for given rollout
        effective_len = self.len_used[0:self.ptr_traj] - (self.traj_rollout - 1)
        size_all = np.sum(effective_len)
        # random indexes in usable range
        idxs = np.random.randint(0, size_all, size=batch_size)
        # adjusted cumulative length for indexs
        cum_len_idxs = np.cumsum(effective_len) - 1
        # calculating rows and columns for usable arrays
        rows = np.searchsorted(cum_len_idxs, idxs)
        cols = idxs - (np.append([0], (cum_len_idxs + 1))[rows])
        return rows, cols

    def sample_trajectories(self, rows, cols):
        cols = tuple([cols + i for i in range(self.traj_rollout)])
        batch = dict(
            state=self.s_buf[rows, cols],
            action=self.action_buf[rows, cols],
            reward=self.rew_buf[rows, cols],
            pi_logp=self.pi_logp_buf[rows, cols],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32,
                                   device=self.device) for k, v in batch.items()}

    def sample_batch(self, batch_size=768):
        effective_len = self.len_used[0:self.ptr_traj]
        size_all = np.sum(effective_len)
        # random indexes in usable range
        idxs = np.random.randint(0, size_all, size=batch_size)
        # adjusted cumulative length for indexs
        cum_len_idxs = np.cumsum(effective_len) - 1
        # calculating rows and columns for usable arrays
        rows = np.searchsorted(cum_len_idxs, idxs)
        cols = idxs - (np.append([0], (cum_len_idxs + 1))[rows])

        batch = dict(
            state=self.s_buf[rows, cols],
            state_next=self.s_next_buf[rows, cols],
            action=self.action_buf[rows, cols],
            reward=self.rew_buf[rows, cols],
            pi_logp=self.pi_logp_buf[rows, cols],
            done=self.done_buf[rows, cols],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32,
                                   device=self.device) for k, v in batch.items()}

    def sample_traj(self):
        row = np.random.randint(0, self.size_traj)
        len = self.len_used[row]
        batch = dict(
            state=self.s_buf[row, 0:len],
            action=self.action_buf[row, 0:len],
            reward=self.rew_buf[row, 0:len],
            pi_logp=self.pi_logp_buf[row, 0:len],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32,
                                   device=self.device).unsqueeze(1) for k, v in batch.items()}

    def stored_interactions(self):
        return self.interactions
