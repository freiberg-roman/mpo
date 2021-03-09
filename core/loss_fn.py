import itertools
import torch


def gaussian_kl(targ_mean, mean, targ_std, std):
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
    max_q = torch.max(targ_q, dim=0).values
    return eta * eps_dual + torch.mean(max_q) \
           + eta * torch.mean(torch.log(torch.mean(torch.exp((targ_q - max_q) / eta), dim=0)))


class PolicyUpdateNonParametric:
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
        qij = torch.softmax(targ_q / self.eta.item(), dim=0)  # (M, B) or (da, B)
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
        loss_eta = dual(self.eta, targ_q, self.eps_dual)
        loss_l = -(
                loss_p
                + eta_mean * (self.eps_mean - c_mean)
                + eta_cov * (self.eps_cov - c_cov)
        ) + loss_eta

        self.writer.add_scalar('combined_loss', loss_l.item() - loss_eta.item(), self.run)
        self.writer.add_scalar('eta', self.eta.item(), self.run)
        self.writer.add_scalar('eta_loss', loss_eta.item(), self.run)
        self.writer.add_scalar('c_mean', c_mean.item(), self.run)
        self.writer.add_scalar('c_cov', c_cov.item(), self.run)

        loss_l.backward()
        self.optimizer.step()
        self.run += 1


class PolicyUpdateParametric:
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
        adv = targ_q - targ_q.mean(dim=0) # (M, B) or (da, B)
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

        # learn eta together with other policy parameters
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
    def __init__(self,
                 writer,
                 eta_mean,
                 eta_cov,
                 eps_mean,
                 eps_cov,
                 lr_kl):
        self.writer = writer
        self.optimizer = torch.optim.Adam(
            itertools.chain([eta_mean], [eta_cov]), lr=lr_kl)
        self.eps_mean = eps_mean
        self.eps_cov = eps_cov
        self.eta_mean = eta_mean
        self.eta_cov = eta_cov

    def __call__(self, c_mean, c_cov, run):
        self.optimizer.zero_grad()

        self.eta_mean.clamp(min=0.0)
        self.eta_cov.clamp(min=0.0)
        loss_eta_mean = self.eta_mean * (self.eps_mean - c_mean.item())
        loss_eta_cov = self.eta_cov * (self.eps_cov - c_cov.item())
        loss_lagrange = loss_eta_mean + loss_eta_cov
        loss_lagrange.backward()

        self.optimizer.step()

        eta_mean_ret = self.eta_mean.item()
        if eta_mean_ret < 0.0:
            eta_mean_ret = 0.0
        self.writer.add_scalar('eta_mean', eta_mean_ret, run)

        eta_cov_ret = self.eta_cov.item()
        if eta_cov_ret < 0.0:
            eta_cov_ret = 0.0
        self.writer.add_scalar('eta_cov', eta_cov_ret, run)

        return eta_mean_ret, eta_cov_ret


class PolicySACUpdate:

    def __init__(self,
                 writer,
                 buffer,
                 ac,
                 actor_optimizer,
                 entropy,
                 batch_size):
        self.writer = writer
        self.buffer = buffer
        self.ac = ac
        self.actor_optimizer = actor_optimizer
        self.entropy = entropy
        self.batch_size = batch_size
        self.run = 0

    def __call__(self):
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