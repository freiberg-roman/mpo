import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import torch.nn.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class GaussianMLPActor(nn.Module):

    def __init__(self, env, hidden_sizes, activation):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.env = env
        self.net = mlp([state_dim] + list(hidden_sizes), activation, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, state):
        net_out = self.net(state)
        mean = self.mean_layer(net_out)
        std = self.std_layer(net_out)
        cov = F.softplus(std) ** 2
        return mean, cov

    def get_logp(self, mean, cov, pi_action):
        # invert scaling
        pi_action = torch.atanh(pi_action / self.act_limit)

        pi_dist = Normal(mean, torch.sqrt(cov))
        logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)  # sum up by action dimension
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)

        return logp_pi

    def get_act(self, mean,  # (..., act_dim)
                cov, deterministic=False):
        pi_dist = Normal(mean, torch.sqrt(cov))
        if deterministic:
            pi_action = mean
            return torch.tanh(pi_action) * self.act_limit, None
        else:
            pi_action = pi_dist.rsample()
            logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)  # sum up by action dimension
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, env, hidden_sizes, activation):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.q = mlp([state_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, state, act):
        q = self.q(torch.cat([state, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, env, hidden_sizes_pi=(100, 100),
                 hidden_sizes_q=(200, 200), activation=nn.ReLU):
        super().__init__()

        self.pi = GaussianMLPActor(env, hidden_sizes_pi, activation)
        self.q1 = MLPQFunction(env, hidden_sizes_q, activation)
        self.q2 = MLPQFunction(env, hidden_sizes_q, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, cov = self.pi(torch.squeeze(obs))
            a, logp_pi = self.pi.get_act(mu, cov, deterministic=deterministic)
            return a.cpu().numpy(), logp_pi
