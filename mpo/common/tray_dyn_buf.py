import numpy as np
import torch


class DynamicTrajectoryBuffer:
    """
    Replay buffer that uses a table to distinct between different trajectories.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        min_rollout,
        max_rollout,
        traj_rollout,
        traj_size,
        device,
    ):
        """
        Initialize DynamicTrajectoryBuffer

        @param state_dim: state dimension defined by the environment
        @param action_dim: action dimension defined by actor from the environment
        @param min_rollout: minimal length of trajectory (mostly environment dependent)
        @param max_rollout: maximal length of trajectory (mostly environment dependent)
        @param traj_rollout: length of rollout used in batches
        @param traj_size: maximum amount of stored trajectories
        @param device: either 'cuda:0' or 'cpu'
        """

        self.s_buf = np.zeros((traj_size, max_rollout, state_dim), dtype=np.float32)
        self.s_next_buf = np.zeros(
            (traj_size, max_rollout, state_dim), dtype=np.float32
        )
        self.action_buf = np.zeros(
            (traj_size, max_rollout, action_dim), dtype=np.float32
        )
        self.rew_buf = np.zeros((traj_size, max_rollout, 1), dtype=np.float32)
        self.pi_logp_buf = np.zeros((traj_size, max_rollout, 1), dtype=np.float32)
        self.done_buf = np.zeros((traj_size, max_rollout, 1), dtype=np.float32)

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
        """
        Stores a complete step with actor information.

        @param state: current state from the environment
        @param state_next: next state from the environment
        @param action:  action performed by the actor in the environment
        @param reward: reward from performing (s, act) pair
        @param pi_logp: logarithmic probability density form actor performing this step
        @param done: 1 or 0 depending if this step is the final step of the trajectory
        """

        assert self.ptr_step < self.max_rollout

        self.s_buf[self.ptr_traj, self.ptr_step, :] = state
        self.s_next_buf[self.ptr_traj, self.ptr_step, :] = state_next
        self.action_buf[self.ptr_traj, self.ptr_step, :] = action
        self.rew_buf[self.ptr_traj, self.ptr_step, :] = reward
        self.pi_logp_buf[self.ptr_traj, self.ptr_step, :] = pi_logp
        self.done_buf[self.ptr_traj, self.ptr_step, :] = done

        self.ptr_step += 1
        self.interactions += 1
        self.len_used[self.ptr_traj] = self.ptr_step

    def next_traj(self):
        """
        This will commit the current trajectory and start a new line for storage.
        """

        assert self.ptr_step >= self.min_rollout  # is defined by environment
        self.len_used[self.ptr_traj] = self.ptr_step  # length of this trajectory
        self.ptr_traj = (self.ptr_traj + 1) % self.max_traj
        self.ptr_step = 0
        self.size_traj = min(self.size_traj + 1, self.max_traj)

    def sample_trajectories(self, batch_size=128):
        """
        Returns a random batch of trajectories with rollout length specified by traj_rollout.

        @param batch_size: size of returned batch
        @return: random batch of size batch_size
        """

        if self.ptr_step > 0:
            # to include current yet unfinished (current) trajectory
            effective_len = self.len_used[
                0 : min(self.size_traj + 1, self.max_traj)
            ] - (self.traj_rollout - 1)
        else:
            effective_len = self.len_used[0 : self.size_traj] - (self.traj_rollout - 1)
        size_all = np.sum(effective_len)

        # random indexes in usable range
        idxs = np.random.randint(0, size_all, size=batch_size)
        cum_len_idxs = np.cumsum(effective_len) - 1

        rows = np.searchsorted(cum_len_idxs, idxs)
        cols = idxs - (np.append([0], (cum_len_idxs + 1))[rows])
        cols = tuple([cols + i for i in range(self.traj_rollout)])

        batch = dict(
            state=self.s_buf[rows, cols],
            action=self.action_buf[rows, cols],
            reward=self.rew_buf[rows, cols],
            pi_logp=self.pi_logp_buf[rows, cols],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def sample_batch(self, batch_size=768):
        """
        Function returns a random batch from stored interactions.

        @param batch_size: size of the batch
        @return: random batch of size batch_size
        """

        effective_len = self.len_used[0 : self.max_traj]
        size_all = np.sum(effective_len)
        # random indexes in usable range
        idxs = np.random.randint(0, size_all, size=batch_size)
        cum_len_idxs = np.cumsum(effective_len) - 1

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
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in batch.items()
        }

    def sample_traj(self):
        """
        Returns a random trajectory form buffer of full length.

        @return: a whole trajectory from buffer
        """

        row = np.random.randint(0, self.size_traj)
        len = self.len_used[row]
        batch = dict(
            state=self.s_buf[row, 0:len],
            action=self.action_buf[row, 0:len],
            reward=self.rew_buf[row, 0:len],
            pi_logp=self.pi_logp_buf[row, 0:len],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device).unsqueeze(1)
            for k, v in batch.items()
        }

    def stored_interactions(self):
        """
        Returns the number of stored steps in this buffer.

        @return: amount of stored steps
        """

        return self.interactions
