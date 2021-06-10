import hydra
import omegaconf

from mpo.eval.mpo_retrace import set_up_mpo_retace


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    if cfg.algorithm.name == "mpo" and cfg.q_learning.name == "retrace":
        set_up_mpo_retace(cfg)
    if cfg.algorithm.name == "mpo" and cfg.q_learning.name == "td0":
        pass
    if cfg.algorithm.name == "mpo_retrace" and cfg.q_learning.name == "retrace":
        pass
    if cfg.algorithm.name == "mpo_retrace" and cfg.q_learning.name == "td0":
        pass
    if cfg.algorithm.name == "sac":
        pass


if __name__ == "__main__":
    run()
