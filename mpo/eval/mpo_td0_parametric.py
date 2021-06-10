from mpo.core.builder.builder_mpo_parametric import mpo_parametric_td0
from torch.utils.tensorboard import SummaryWriter


def set_up_mpo_parametric_td0(cfg):
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

        mpo_parametric_td0(
            env_name=cfg.overrides.env,
            local_device=cfg.device,
            writer=writer,
            lr_pi=cfg.algorithm.lr_pi,
            lr_q=cfg.algorithm.lr_q,
            lr_kl=cfg.algorithm.lr_kl,
            eps_mean=cfg.algorithm.eps_mean,
            eps_cov=cfg.algorithm.eps_cov,
            gamma=cfg.algorithm.gamma,
            batch_size=cfg.algorithm.batch_state,
            batch_size_act=cfg.algorithm.batch_action,
            update_steps=cfg.algorithm.update_steps,
            update_after=cfg.algorithm.update_after,
            min_steps_per_epoch=cfg.algorithm.min_steps_per_epoch,
            test_after=cfg.algorithm.test_after,
            total_steps=cfg.overrides.total_steps,
        )()
