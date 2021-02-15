import itertools
import torch
from mpo_mod.core import GaussianMLPActor
from common.retrace import Retrace


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


class UpdateQ_TD0:
    def __init__(self,
                 writer,
                 critic_optimizer,
                 ac,
                 ac_targ,
                 buffer,
                 batch_size,
                 gamma,
                 entropy):
        self.writer = writer
        self.critic_optimizer = critic_optimizer
        self.ac = ac
        self.ac_targ = ac_targ
        self.buffer = buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy = entropy
        self.run = 0
        self.polyak = 0.995

    def __call__(self):
        self.critic_optimizer.zero_grad()

        samples = self.buffer.sample_batch(batch_size=self.batch_size)

        q1 = self.ac.q1(samples['state'], samples['action'])
        q2 = self.ac.q2(samples['state'], samples['action'])

        with torch.no_grad():
            mean, std = self.ac.pi.forward(samples['state_next'])
            # mean, std = self.ac_targ.pi.forward(samples['state_next'])
            act_next = GaussianMLPActor.get_act(mean, std)
            logp = GaussianMLPActor.get_logp(mean, std, act_next)

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

        for p, p_targ in zip(self.ac.q1.parameters(), self.ac_targ.q1.parameters()):
            p_targ.data.mul_(self.polyak)
            p_targ.data.add_((1 - self.polyak) * p.data)

        for p, p_targ in zip(self.ac.q2.parameters(), self.ac_targ.q2.parameters()):
            p_targ.data.mul_(self.polyak)
            p_targ.data.add_((1 - self.polyak) * p.data)

        self.run += 1


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
            sampled_actions = GaussianMLPActor.get_act(b_μ, b_A, amount=(M,))
            expanded_states = state_batch[None, ...].expand(M, -1, -1)  # (M, B, ds)
            targ_q1 = self.ac_targ.q1.forward(
                expanded_states.reshape(-1, self.ds),  # (M * B, ds)
                sampled_actions.reshape(-1, self.da)  # (M * B, da)
            ).reshape(M, B)  # (M, B)
            targ_q2 = self.ac_targ.q2.forward(
                expanded_states.reshape(-1, self.ds),  # (M * B, ds)
                sampled_actions.reshape(-1, self.da)  # (M * B, da)
            ).reshape(M, B)  # (M, B)
            targ_q = torch.min(targ_q1, targ_q2)

        # M-step
        qij = torch.softmax(targ_q / self.eta.item(), dim=0)  # (M, B) or (da, B)
        # update policy based on lagrangian
        mean, std = self.ac.pi.forward(state_batch)
        loss_p = torch.mean(
            qij * (
                    GaussianMLPActor.get_logp(mean, b_A, sampled_actions, expand=(M, B)) +
                    GaussianMLPActor.get_logp(b_μ, std, sampled_actions, expand=(M, B))
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
        loss_lagrauge = loss_eta_mean + loss_eta_cov
        loss_lagrauge.backward()

        self.optimizer.step()

        eta_mean_ret = self.eta_mean.item()
        if eta_mean_ret < 0.0:
            eta_mean_ret = 0.0
        self.writer.add_scalar('eta_mean', eta_mean_ret, run)

        eta_cov_ret = self.eta_cov.item()
        if eta_cov_ret < 0.0:
            eta_cov_ret = 0.0
        self.writer.add_scalar('eta_cov', self.eta_cov, run)

        return eta_mean_ret, eta_cov_ret
