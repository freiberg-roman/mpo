from mpo_rerpi_style import core
from mpo_rerpi_style.mpo_td0 import mpo_td
from utils.report import create_graphs
import argparse
import torch
import gym

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
    parser.add_argument('--exp_name', type=str, default='mpo_soft')
    args = parser.parse_args()

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name,
                                        args.seed,
                                        "../results/",
                                        True)

    torch.set_num_threads(torch.get_num_threads())
    mpo_td(lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes_q=[args.hid_q] * args.l,
                       hidden_sizes_pi=[args.hid_pi] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        reward_scaling=lambda r: r,
        )

    # create_graphs(logger_kwargs['output_dir'] + '/progress.txt', save_to=logger_kwargs['output_dir'],
    #               plot=[('Epoch', 'AverageTestEpRet'),
    #                     ('Epoch', 'AverageEta')])
