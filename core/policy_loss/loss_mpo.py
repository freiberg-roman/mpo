import itertools
import torch
import torch.nn.functional as F


def gaussian_kl(targ_mean, mean, targ_std, std):
    """
    Computes the gaussian KL for both mean and covariance. Covariance is assumed to be a diagonal
    matrix.

    B stands for batch size
    A stands for action dimension (defined by environment)
    @param targ_mean: (B, A) tensor with mean values from target network
    @param mean: (B, A) tensor with mean values from current network
    @param targ_std: (B, A) tensor with standard deviations from target network
    @param std: (B, A) tensor with standard deviations from current network
    @return: tuple of mean -and covariance KL as a tensor
    """

    n = std.size(-1)
    cov = std ** 2
    targ_cov = targ_std ** 2
    cov_inv = 1 / cov
    targ_cov_inv = 1 / targ_cov
    inner_mean = (((mean - targ_mean) ** 2) * targ_cov_inv).sum(-1)
    inner_cov = torch.log(cov.prod(-1) / targ_cov.prod(-1)) - n + (cov_inv * targ_cov).sum(-1)
    c_mean = 0.5 * torch.mean(inner_mean)
    c_cov = 0.5 * torch.mean(inner_cov)
    return c_mean, c_cov


def dual(eta, targ_q, eps_dual):
    """
    Computes the loss of the dual function which results from an lagrange optimization problem.
    This is a more numerical stable equivalent then the original proposal.

    @param eta: tensor of current eta value
    @param targ_q: (BA, BS)
    @param eps_dual: float for restriction of dual function
    @return: loss of dual function as a tensor
    """
    # Bin mir nicht 100% sicher aber wenn hier als Argument D := targ_q / eta verwendet
    # dann sollte ja D_max = max_q / eta sein und somit dann auch
    # D - D_max + D_max = (targ - max_q) / eta + max_q / eta gelten. Dann würde aber sich eta rauskürzen.
    # Wenn man aber hier vor torch.mean(max_q) ein eta vorschreibt, funktioniert der Algorithmus auch.
    # Bemerkung: eta ist hier softplus da es im Argument übergeben wird

    max_q = torch.max(targ_q, dim=0).values
    return eta * eps_dual + torch.mean(max_q) \
           + eta * torch.mean(torch.log(torch.mean(torch.exp((targ_q - max_q) / eta), dim=0)))


class PolicyUpdateNonParametric:
    """
    Capsules all the needed data to perform one update cycle per call for the policy.
    This version implements the loss of the non parametric MPO version.
    """

    def __init__(self,
                 device,
                 writer,
                 ac,
                 ac_targ,
                 actor_eta_optimizer,
                 eta,
                 eps_mean,
                 eps_cov,
                 eps_dual,
                 lr_kl,
                 buffer,
                 batch_size,
                 batch_size_act,
                 ds,
                 da):
        """
        Initializes all values that are needed for an update cycle of the policy

        @param device: device either 'cuda:0' or 'cpu'
        @param writer: Summary writer from tensorboard or an equivalent stub
        @param ac: current actor critic networks
        @param ac_targ: target actor critic networks
        @param actor_eta_optimizer: Adam of similar one step optimizer for the policy network
        @param eta: tensor with the initial eta value
        @param eps_mean: epsilon for the mean kl constrain
        @param eps_cov: epsilon for the covariance kl constrain
        @param eps_dual: epsilon for the dual function constrain
        @param lr_kl: learning rate for the kl values
        @param buffer: replay buffer that provides samples from performed trajectories
        @param batch_size: batch size for states
        @param batch_size_act: batch size for actions per state
        @param ds: dimension of state space (defined by the environment)
        @param da: dimension of action space (defined by the environment)
        """

        self.writer = writer
        self.ac = ac
        self.ac_targ = ac_targ
        self.optimizer = actor_eta_optimizer
        self.eta = eta
        self.eps_mean = eps_mean
        self.eps_cov = eps_cov
        self.eps_dual = eps_dual
        self.buffer = buffer
        self.B = batch_size
        self.M = batch_size_act
        self.ds = ds
        self.da = da
        self.update_lagrange = UpdateLagrangeTrustRegionOptimizer(
            writer,
            torch.tensor([0.0], device=device, requires_grad=True),
            torch.tensor([0.0], device=device, requires_grad=True),
            eps_mean,
            eps_cov,
            lr_kl
        )
        self.run = 0

    def __call__(self):
        """
        Performs an update step for the policy
        """

        B = self.B
        M = self.M
        samples = self.buffer.sample_batch(batch_size=B)
        state_batch = samples['state']
        with torch.no_grad():
            b_μ, b_A = self.ac_targ.pi.forward(state_batch)  # (B,)
            sampled_actions = self.ac_targ.get_act(b_μ, b_A, amount=(M,))
            expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
            targ_q = self.ac_targ.q_forward(
                expanded_states.reshape(-1, self.ds),  # (M * B, ds)
                sampled_actions.reshape(-1, self.da)  # (M * B, da)
            ).reshape(M, B)

        # M-step
        qij = torch.softmax(targ_q / F.softplus(self.eta).item(), dim=0)  # (M, B) or (da, B)
        # update policy based on lagrangian
        mean, std = self.ac.pi.forward(state_batch)
        loss_p = torch.mean(
            qij * (
                    self.ac_targ.get_logp(mean, b_A, sampled_actions, expand=(M, B)) +
                    self.ac_targ.get_logp(b_μ, std, sampled_actions, expand=(M, B))
            )
        )
        self.writer.add_scalar('loss_pi', loss_p.item(), self.run)

        c_mean, c_cov = gaussian_kl(
            targ_mean=b_μ, mean=mean,
            targ_std=b_A, std=std)

        # Update lagrange multipliers by gradient descent
        eta_mean, eta_cov = self.update_lagrange(c_mean, c_cov, self.run)

        # learn eta together with other policy parameters
        self.optimizer.zero_grad()
        loss_eta = dual(F.softplus(self.eta), targ_q, self.eps_dual)
        loss_l = -(
                loss_p
                + eta_mean * (self.eps_mean - c_mean)
                + eta_cov * (self.eps_cov - c_cov)
        ) + loss_eta

        self.writer.add_scalar('combined_loss', loss_l.item() - loss_eta.item(), self.run)
        self.writer.add_scalar('eta', F.softplus(self.eta).item(), self.run)
        self.writer.add_scalar('eta_loss', loss_eta.item(), self.run)
        self.writer.add_scalar('c_mean', c_mean.item(), self.run)
        self.writer.add_scalar('c_cov', c_cov.item(), self.run)

        loss_l.backward()
        self.optimizer.step()
        self.run += 1


class PolicyUpdateParametric:
    """
    Capsules all the needed data to perform one update cycle per call for the policy.
    This version implements the loss of the parametric MPO version.
    """

    def __init__(self,
                 device,
                 writer,
                 ac,
                 ac_targ,
                 actor_eta_optimizer,
                 eps_mean,
                 eps_cov,
                 lr_kl,
                 buffer,
                 batch_size,
                 batch_size_act,
                 ds,
                 da):
        """
        Initializes all values that are needed for an update cycle of the policy

        @param device: device either 'cuda:0' or 'cpu'
        @param writer: Summary writer from tensorboard or an equivalent stub
        @param ac: current actor critic networks
        @param ac_targ: target actor critic networks
        @param actor_eta_optimizer: Adam of similar one step optimizer for the policy network
        @param eps_mean: epsilon for the mean kl constrain
        @param eps_cov: epsilon for the covariance kl constrain
        @param lr_kl: learning rate for the kl values
        @param buffer: replay buffer that provides samples from performed trajectories
        @param batch_size: batch size for states
        @param batch_size_act: batch size for actions per state
        @param ds: dimension of state space (defined by the environment)
        @param da: dimension of action space (defined by the environment)
        """

        self.writer = writer
        self.ac = ac
        self.ac_targ = ac_targ
        self.optimizer = actor_eta_optimizer
        self.eps_mean = eps_mean
        self.eps_cov = eps_cov
        self.buffer = buffer
        self.B = batch_size
        self.M = batch_size_act
        self.ds = ds
        self.da = da
        self.update_lagrange = UpdateLagrangeTrustRegionOptimizer(
            writer,
            torch.tensor([0.0], device=device, requires_grad=True),
            torch.tensor([0.0], device=device, requires_grad=True),
            eps_mean,
            eps_cov,
            lr_kl
        )
        self.run = 0

    def __call__(self):
        """
        Performs an update step for the policy
        """

        B = self.B
        M = self.M
        samples = self.buffer.sample_batch(batch_size=B)
        state_batch = samples['state']
        with torch.no_grad():
            b_μ, b_A = self.ac_targ.pi.forward(state_batch)  # (B,)
            sampled_actions = self.ac_targ.get_act(b_μ, b_A, amount=(M,))
            expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
            targ_q = self.ac_targ.q_forward(
                expanded_states.reshape(-1, self.ds),  # (M * B, ds)
                sampled_actions.reshape(-1, self.da)  # (M * B, da)
            ).reshape(M, B)

        # M-step
        adv = targ_q - targ_q.mean(dim=0)  # (M, B) or (da, B)
        # update policy based on lagrangian
        mean, std = self.ac.pi.forward(state_batch)
        loss_p = torch.mean(
            adv * (
                    self.ac_targ.get_logp(mean, b_A, sampled_actions, expand=(M, B)) +
                    self.ac_targ.get_logp(b_μ, std, sampled_actions, expand=(M, B))
            )
        )
        self.writer.add_scalar('loss_pi', loss_p.item(), self.run)

        c_mean, c_cov = gaussian_kl(
            targ_mean=b_μ, mean=mean,
            targ_std=b_A, std=std)

        # Update lagrange multipliers by gradient descent
        eta_mean, eta_cov = self.update_lagrange(c_mean, c_cov, self.run)

        self.optimizer.zero_grad()
        loss_l = -(
                loss_p
                + eta_mean * (self.eps_mean - c_mean)
                + eta_cov * (self.eps_cov - c_cov)
        )

        self.writer.add_scalar('combined_loss', loss_l.item(), self.run)
        self.writer.add_scalar('c_mean', c_mean.item(), self.run)
        self.writer.add_scalar('c_cov', c_cov.item(), self.run)

        loss_l.backward()
        self.optimizer.step()
        self.run += 1


class UpdateLagrangeTrustRegionOptimizer:
    """
    Capsules all the data required to perform an update for the KL values in the policy update step.
    """

    def __init__(self,
                 writer,
                 eta_mean,
                 eta_cov,
                 eps_mean,
                 eps_cov,
                 lr_kl):
        """
        Initializes all values that are needed for an update cycle of KL values

        @param writer: Summary writer from tensorboard or an equivalent stub
        @param eta_mean: tensor with initial value for lagrange mean
        @param eta_cov: tensor with initial value for lagrange covariance
        @param eps_mean: epsilon for the mean kl constrain
        @param eps_cov: epsilon for the covariance kl constrain
        @param lr_kl: learning rate for the kl values
        """

        self.writer = writer
        self.optimizer = torch.optim.Adam(
            itertools.chain([eta_mean], [eta_cov]), lr=lr_kl)
        self.eps_mean = eps_mean
        self.eps_cov = eps_cov
        self.eta_mean = eta_mean
        self.eta_cov = eta_cov

    def __call__(self, c_mean, c_cov, run):
        """
        Performs an update step for the KL values

        @param c_mean: mean KL value
        @param c_cov: covariance KL value
        @param run: current run of the update cycle
        @return: lagrange loss for mean and covariance
        """

        self.optimizer.zero_grad()

        loss_eta_mean = F.softplus(self.eta_mean) * (self.eps_mean - c_mean.item())
        loss_eta_cov = F.softplus(self.eta_cov) * (self.eps_cov - c_cov.item())
        loss_lagrange = loss_eta_mean + loss_eta_cov
        loss_lagrange.backward()

        self.optimizer.step()

        self.writer.add_scalar('eta_mean', F.softplus(self.eta_mean), run)
        self.writer.add_scalar('eta_cov', F.softplus(self.eta_cov), run)

        return F.softplus(self.eta_mean), F.softplus(self.eta_cov)
