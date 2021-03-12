import torch
from common.retrace import Retrace


class UpdateQRetrace:
    """
    Encapsulates Retrace algorithm for Q function updates such that on call one update cycle is performed.
    """

    def __init__(self,
                 writer,
                 critic_optimizer,
                 ac,
                 ac_targ,
                 buffer,
                 batch_size,
                 gamma,
                 device):
        """
        Initializes needed values

        @param writer: SummaryWriter from tensorboard for logging
        @param critic_optimizer: Adam or similar one step optimizer for Q function updates
        @param ac: current actor critic model
        @param ac_targ: target actor critic model
        @param buffer: replay buffer capable of sampling trajectories
        @param batch_size: size of batch for retrace algorithm
        @param gamma: discount factor
        @param device: either 'cuda:0' or 'cpu'
        """

        self.writer = writer
        self.critic_optimizer = critic_optimizer
        self.ac = ac
        self.ac_targ = ac_targ
        self.buffer = buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.run = 0
        self.polyak = 0.995

    def __call__(self):
        """
        On call whole update cycle is performed.
        """

        self.critic_optimizer.zero_grad()

        samples = self.buffer.sample_trajectories(batch_size=self.batch_size)
        batch_q = self.ac.q.forward(samples['state'], samples['action'])
        batch_q = torch.transpose(batch_q, 0, 1)

        targ_q = self.ac_targ.q.forward(samples['state'], samples['action'])
        targ_q = torch.transpose(targ_q, 0, 1)

        targ_mean, targ_chol = self.ac_targ.pi.forward(samples['state'])
        targ_act = self.ac_targ.get_act(targ_mean, targ_chol)

        exp_targ_q = self.ac_targ.q.forward(samples['state'], targ_act)
        exp_targ_q = torch.transpose(exp_targ_q, 0, 1)

        targ_act_logp = self.ac_targ.get_logp(targ_mean, targ_chol, samples['action']).unsqueeze(-1)
        targ_act_logp = torch.transpose(targ_act_logp, 0, 1)

        retrace = Retrace(device=self.device)
        loss_q = retrace(Q=batch_q,
                         expected_target_Q=exp_targ_q,
                         target_Q=targ_q,
                         rewards=torch.transpose(samples['reward'], 0, 1),
                         target_policy_probs=targ_act_logp,
                         behaviour_policy_probs=torch.transpose(samples['pi_logp'], 0, 1),
                         gamma=self.gamma
                         )
        self.writer.add_scalar('q_loss', loss_q.item(), self.run)
        self.writer.add_scalar('q', targ_q.detach().mean().item(), self.run)
        self.writer.add_scalar('q_min', targ_q.detach().min().item(), self.run)
        self.writer.add_scalar('q_max', targ_q.detach().max().item(), self.run)

        loss_q.backward()
        self.critic_optimizer.step()

        self.run += 1


class UpdateQ_TD:
    """
    Encapsulates TD0 algorithm for Q function updates such that on call one update cycle is performed.
    """

    def __init__(self,
                 writer,
                 critic_optimizer,
                 ac,
                 ac_targ,
                 buffer,
                 batch_size,
                 gamma,
                 entropy):
        """
        Initializes needed values

        @param writer: SummaryWriter from tensorboard for logging
        @param critic_optimizer: Adam or similar one step optimizer for Q function updates
        @param ac: current actor critic model
        @param ac_targ: target actor critic model
        @param buffer: replay buffer capable of sampling trajectories
        @param batch_size: size of batch for retrace algorithm
        @param gamma: discount factor
        @param entropy: steers the influence of entropy in estimation
        """

        self.writer = writer
        self.critic_optimizer = critic_optimizer
        self.ac = ac
        self.ac_targ = ac_targ
        self.buffer = buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy = entropy
        self.run = 0

    def __call__(self):
        """
        On call whole update cycle is performed.
        """

        self.critic_optimizer.zero_grad()

        samples = self.buffer.sample_batch(batch_size=self.batch_size)

        q1 = self.ac.q1(samples['state'], samples['action'])
        q2 = self.ac.q2(samples['state'], samples['action'])

        with torch.no_grad():
            mean, std = self.ac.pi.forward(samples['state_next'])
            act_next = self.ac.get_act(mean, std)
            logp = self.ac.get_logp(mean, std, act_next)

            targ_q1 = self.ac_targ.q1(samples['state_next'], act_next)
            targ_q2 = self.ac_targ.q2(samples['state_next'], act_next)
            targ_q = torch.min(targ_q1, targ_q2)
            backup = samples['reward'] + \
                     self.gamma * (1 - samples['done']) * (targ_q - self.entropy * logp)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        self.writer.add_scalar('q_loss', loss_q.item(), self.run)
        self.writer.add_scalar('q', targ_q.detach().mean().item(), self.run)
        self.writer.add_scalar('q_min', targ_q.detach().min().item(), self.run)
        self.writer.add_scalar('q_max', targ_q.detach().max().item(), self.run)

        loss_q.backward()
        self.critic_optimizer.step()

        self.run += 1


class UpdateQ_TDE:
    """
    Implements Soft Actor Critic Q update function such that on call a whole update cycle is performed.
    """

    def __init__(self,
                 writer,
                 critic_optimizer,
                 ac,
                 ac_targ,
                 buffer,
                 batch_size,
                 gamma,
                 entropy):
        """
        Initializes needed values

        @param writer: SummaryWriter from tensorboard for logging
        @param critic_optimizer: Adam or similar one step optimizer for Q function updates
        @param ac: current actor critic model
        @param ac_targ: target actor critic model
        @param buffer: replay buffer capable of sampling trajectories
        @param batch_size: size of batch for retrace algorithm
        @param gamma: discount factor
        @param entropy: steers the influence of entropy in estimation
        """

        self.writer = writer
        self.critic_optimizer = critic_optimizer
        self.ac = ac
        self.ac_targ = ac_targ
        self.buffer = buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy = entropy
        self.run = 0

    def __call__(self):
        """
        On call whole update cycle is performed.
        """

        self.critic_optimizer.zero_grad()
        samples = self.buffer.sample_batch(batch_size=self.batch_size)

        q1 = self.ac.q1(samples['state'], samples['action'])
        q2 = self.ac.q2(samples['state'], samples['action'])

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            act_next, logp_act_next = self.ac.pi.forward(samples['state_next'])

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(samples['state_next'], act_next)
            q2_pi_targ = self.ac_targ.q2(samples['state_next'], act_next)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = samples['reward'] + self.gamma * (1 - samples['done']) * (q_pi_targ - self.entropy * logp_act_next)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        self.writer.add_scalar('q_loss', loss_q.item(), self.run)
        self.writer.add_scalar('q', torch.min(q1.detach(), q2.detach()).cpu().numpy().mean(), self.run)
        self.writer.add_scalar('q_min', torch.min(q1.detach(), q2.detach()).cpu().numpy().min(), self.run)
        self.writer.add_scalar('q_max', torch.min(q1.detach(), q2.detach()).cpu().numpy().max(), self.run)

        loss_q.backward()
        self.critic_optimizer.step()

        self.run += 1
