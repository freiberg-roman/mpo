import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


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

    def __init__(self, state_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([state_dim] + list(hidden_sizes), activation, activation)

        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, state):
        net_out = self.net(state)
        mu = self.mu_layer(net_out)
        std = self.std_layer(net_out)
        soft_plus_std = torch.log(torch.exp(std) + 1)
        covariance = soft_plus_std ** 2
        return mu, covariance

    def get_prob(self, mu, covariance, action):
        pi_distribution = Normal(mu, covariance)
        logp = pi_distribution.log_prob(action)
        return logp

    def get_act(self, mu, covariance, n, deterministic=False, traj=False, batch=False):
        mu = mu.clone()
        covariance = covariance.clone()

        if deterministic:
            return mu.repeat(n, 1), None

        if traj:
            mu = torch.reshape(mu, (1, 200, 1))  # I am tiered !!!!! must rework !!!!!!
            covariance = torch.reshape(covariance, (1, 200, 1))  # I am tiered !!!!! must rework !!!!!!
            mu = mu.repeat(n, 1, 1)
            covariance = covariance.repeat(n, 1, 1)
        elif batch:
            mu = torch.reshape(mu, (128, 1, 1))  # I am tiered !!!!! must rework !!!!!!
            covariance = torch.reshape(covariance, (128, 1, 1))  # I am tiered !!!!! must rework !!!!!!
            mu = mu.repeat(1, n, 1)
            covariance = covariance.repeat(1, n, 1)
        else:
            mu = mu.repeat(n, 1)
            covariance = covariance.repeat(n, 1)

        pi_distribution = Normal(mu, covariance)
        actions = pi_distribution.rsample()
        return actions, pi_distribution.log_prob(actions)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, state_space, action_space, hidden_sizes_pi=(100, 100),
                 hidden_sizes_q=(200, 200), activation=nn.ReLU):
        super().__init__()

        state_dim = state_space.shape[0]
        action_dim = action_space.shape[0]
        self.pi = GaussianMLPActor(state_dim, action_dim, hidden_sizes_pi, activation).cuda()
        self.q = MLPQFunction(state_dim, action_dim, hidden_sizes_q, activation).cuda()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, cov = self.pi(torch.squeeze(obs))
            a, logp_pi = self.pi.get_act(mu, cov, 1, deterministic=deterministic)
            return a.cpu().numpy(), logp_pi
