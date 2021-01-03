from sac_algorithm import core
from sac_algorithm.sac import sac
import argparse
import torch
import gym

from utils.plot import make_plots
from utils.test_policy import load_policy_and_env, run_policy

if __name__ == "__main__":
    # run sac with needed parameters for 5 epochs

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCar-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name,
                                        args.seed,
                                        "../results/",
                                        True)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)

    # # plot results in graph and safe to results
    make_plots(["../results/"])

    # only for making agent visible
    # env, act = load_policy_and_env("../results/2020-12-21_sac/first_try/")
    # run_policy(env, act)
