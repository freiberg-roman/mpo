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
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class GaussianMLPActor(nn.Module):

    def __init__(self, env, device, hidden_sizes, activation):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.env = env
        self.net = mlp([state_dim] + list(hidden_sizes), activation, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.act_dim = act_dim
        self.device = device

    def forward(self, state):
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(self.device)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(self.device)
        net_out = self.net(state)
        mean = torch.sigmoid(self.mean_layer(net_out))
        mean = action_low + (action_high - action_low) * mean
        std = self.std_layer(net_out)
        std = F.softplus(std)
        return mean, std


class MLPQFunction(nn.Module):

    def __init__(self, env, hidden_sizes, activation):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.q = mlp([state_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, state, act):
        return self.q(torch.cat([state, act], dim=-1))


class MLPActorCritic(nn.Module):

    def __init__(self, env, device, hidden_sizes_pi=(256, 256),
                 hidden_sizes_q=(256, 256), activation=nn.ReLU):
        super().__init__()

        self.pi = GaussianMLPActor(env, device, hidden_sizes_pi, activation)
        self.q1 = MLPQFunction(env, hidden_sizes_q, activation)
        self.q2 = MLPQFunction(env, hidden_sizes_q, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            mu, cov = self.pi(torch.squeeze(obs))
            a, logp_pi = self.pi.get_act(mu, cov, deterministic=deterministic)
            return a.cpu().numpy(), logp_pi

    def q_forward(self, state, act):
        return torch.min(self.q1.forward(state, act),
                         self.q2.forward(state, act))

    @staticmethod
    def get_logp(mean, std, action, expand=None):
        if expand is not None:
            dist = Independent(Normal(mean, std), reinterpreted_batch_ndims=1).expand(expand)
        else:
            dist = Independent(Normal(mean, std), reinterpreted_batch_ndims=1)
        return dist.log_prob(action)

    @staticmethod
    def get_act(mean, std, amount=None):
        dist = Independent(Normal(mean, std), reinterpreted_batch_ndims=1)
        if amount is not None:
            return dist.sample(amount)
        return dist.sample()
