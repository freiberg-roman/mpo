import hydra
import omegaconf

from mpo.eval.mpo_retrace import set_up_mpo_retace
from mpo.eval.mpo_retrace_parametric import set_up_mpo_retace_parametric
from mpo.eval.mpo_td0 import set_up_mpo_td0
from mpo.eval.mpo_td0_parametric import set_up_mpo_parametric_td0
from mpo.eval.sac import set_up_sac


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    if cfg.algorithm.name == "mpo" and cfg.q_learning.name == "retrace":
        set_up_mpo_retace(cfg)
    if cfg.algorithm.name == "mpo" and cfg.q_learning.name == "td0":
        set_up_mpo_td0(cfg)
    if cfg.algorithm.name == "mpo_parametric" and cfg.q_learning.name == "retrace":
        set_up_mpo_retace_parametric(cfg)
    if cfg.algorithm.name == "mpo_parametric" and cfg.q_learning.name == "td0":
        set_up_mpo_parametric_td0(cfg)
    if cfg.algorithm.name == "sac":
        set_up_sac(cfg)


if __name__ == "__main__":
    run()
