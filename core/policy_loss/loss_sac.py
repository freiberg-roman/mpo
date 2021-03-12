import torch


class PolicySACUpdate:
    """
    Capsules all the needed data to perform one update cycle per call for the policy.
    This version implements the loss of Soft Actor Critic
    """

    def __init__(self,
                 writer,
                 buffer,
                 ac,
                 actor_optimizer,
                 entropy,
                 batch_size):
        """
        Initialize values

        @param writer: Summary writer from tensorboard or an equivalent stub
        @param buffer: replay buffer that provides samples from performed trajectories
        @param ac: current actor critic networks
        @param actor_optimizer: Adam of similar one step optimizer for the policy network
        @param entropy: temperature parameter that influences the impact of entropy in the reward
        @param batch_size: batch size for states
        """

        self.writer = writer
        self.buffer = buffer
        self.ac = ac
        self.actor_optimizer = actor_optimizer
        self.entropy = entropy
        self.batch_size = batch_size
        self.run = 0

    def __call__(self):
        """
        Performs an update step for the policy
        """

        self.actor_optimizer.zero_grad()

        samples = self.buffer.sample_batch(batch_size=self.batch_size)
        s = samples['state']
        pi, logp_pi = self.ac.pi.forward(s)

        q1_pi = self.ac.q1(s, pi)
        q2_pi = self.ac.q2(s, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.entropy * logp_pi - q_pi).mean()

        loss_pi.backward()
        self.actor_optimizer.step()

        # Useful info for logging
        self.writer.add_scalar('pi_loss', loss_pi.item(), self.run)
        self.writer.add_scalar('pi_logp', logp_pi.detach().cpu().numpy().mean(), self.run)

        self.run += 1
