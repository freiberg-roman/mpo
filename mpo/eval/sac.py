from mpo.core.sac.sac import sac
import gym
from torch.utils.tensorboard import SummaryWriter


def set_up_sac(cfg):
    for i in range(cfg.repeat):
        if cfg.repeat == 1:
            writer = SummaryWriter(
                "../runs/" + cfg.algorithm.name + "_" + cfg.q_learning.name
            )
        else:
            writer = SummaryWriter(
                "../runs/"
                + cfg.algorithm.name
                + "_"
                + cfg.q_learning.name
                + "_"
                + str(i)
            )
        sac(
            env_fn=lambda: gym.make(cfg.overrides.env),
            writer=writer,
            epochs=cfg.overrides.total_steps // 5000,
        )
