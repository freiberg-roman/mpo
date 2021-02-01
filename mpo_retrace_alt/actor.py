import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorContinuous(nn.Module):
    """
    Policy network
    :param env: OpenAI gym environment
    """
    def __init__(self, env):
        super(ActorContinuous, self).__init__()
        self.env = env
        self.ds = env.observation_space.shape[0]
        self.da = env.action_space.shape[0]
        self.lin1 = nn.Linear(self.ds, 256)
        self.lin2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, self.da)
        self.std_layer = nn.Linear(256, self.da)

    def forward(self, state):
        """
        forwards input through the network
        :param state: (B, ds)
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        device = state.device
        action_low = torch.from_numpy(self.env.action_space.low)[None, ...].to(device)  # (1, da)
        action_high = torch.from_numpy(self.env.action_space.high)[None, ...].to(device)  # (1, da)
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        mean = torch.sigmoid(self.mean_layer(x))  # (B, da)
        mean = action_low + (action_high - action_low) * mean
        std = F.softplus(self.std_layer(x))
        return mean, std
