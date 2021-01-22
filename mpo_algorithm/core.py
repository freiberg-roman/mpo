import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class GaussianMLPActor(nn.Module):

    def __init__(self, env, hidden_sizes, activation):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.env = env
        self.net = mlp([state_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, state):
        act_lim_low = torch.from_numpy(
            # self.env.action_space.low)[None, ...].to("cuda:0")
            self.env.action_space.low)[None, ...]
        act_lim_high = torch.from_numpy(
            # self.env.action_space.high)[None, ...].to("cuda:0")
            self.env.action_space.high)[None, ...]
        net_out = self.net(state)
        mu = torch.sigmoid(self.mu_layer(net_out))
        # enforce bounds on mu
        mu = act_lim_low + (act_lim_high - act_lim_low) * mu
        std = self.std_layer(net_out)
        soft_plus_std = torch.log(torch.exp(std) + 1)
        covariance = soft_plus_std ** 2
        return mu, covariance

    def get_dist(self, mean, cov):
        pi_distribution = Independent(Normal(mean, cov), 1)
        return pi_distribution

    def get_act(self, mean,  # (batch_s, act_dim)
                cov, n, deterministic=False):
        if deterministic:
            return mean.repeat(n, 1), None

        pi_distribution = Independent(Normal(mean, cov), 1)
        actions = pi_distribution.expand((n, mean.shape[0])).rsample()
        return actions, pi_distribution.log_prob(actions)


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
        # self.pi = GaussianMLPActor(env, hidden_sizes_pi, activation).cuda()
        self.q = MLPQFunction(env, hidden_sizes_q, activation)
        # self.q = MLPQFunction(env, hidden_sizes_q, activation).cuda()

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, cov = self.pi(torch.squeeze(obs))
            a, logp_pi = self.pi.get_act(mu, cov, 1, deterministic=deterministic)
            return a.cpu().numpy(), logp_pi
