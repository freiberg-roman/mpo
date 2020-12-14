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

        # layer outputs the mean of action
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        # layer outputs log cholensky matrix in vector form which is used to
        # calculate covariance of policy
        self.std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, state, deterministic=False, with_logprob=True):
        # parameters from net
        net_out = self.net(state)
        mu = self.mu_layer(net_out)
        std = self.std_layer(net_out)
        soft_plus_std = torch.log(torch.exp(std) + 1)

        # distribution
        pi_distribution = Normal(mu, soft_plus_std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action)
        else:
            logp_pi = None

        return pi_action, logp_pi, mu, soft_plus_std

    def get_prob(self, state, action):
        # parameters from net
        net_out = self.net(state)
        mu = self.mu_layer(net_out)
        std = self.std_layer(net_out)
        soft_plus_std = torch.log(torch.exp(std) + 1)

        # distribution
        pi_distribution = Normal(mu, soft_plus_std)
        return pi_distribution.log_prob(action)



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
        self.pi = GaussianMLPActor(state_dim, action_dim, hidden_sizes_pi, activation)
        self.q = MLPQFunction(state_dim, action_dim, hidden_sizes_q, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, logp_pi, _, _ = self.pi(obs, deterministic, True)
            return a.numpy(), logp_pi
