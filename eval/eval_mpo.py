from mpo_algorithm import core
from mpo_algorithm.mpo import mpo
import argparse
import torch
import gym

from utils.plot import make_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    # parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--hid_q', type=int, default=256)
    parser.add_argument('--hid_pi', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--exp_name', type=str, default='mpo')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name,
                                        args.seed,
                                        "../results/",
                                        True)

    torch.set_num_threads(torch.get_num_threads())
    mpo(lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes_q=[args.hid_q] * args.l,
                       hidden_sizes_pi=[args.hid_pi] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        reward_scaling=lambda r: r*0.30725*0.2 + 1,

        logger_kwargs=logger_kwargs
        )

    # plot results in graph and safe to results
    make_plots(["../results/"], yaxis='AverageTestEpRet')
