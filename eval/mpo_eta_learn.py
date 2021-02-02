import argparse
from mpo_eta.mpo_retrace_eta import mpo_retrace
import gym
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--eps_dual', type=float, default=0.1)
    parser.add_argument('--eps_mean', type=float, default=0.1)
    parser.add_argument('--eps_cov', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--sample_episodes', type=int, default=1)
    parser.add_argument('--episode_length', type=int, default=200)
    parser.add_argument('--batch_action', type=int, default=20)
    parser.add_argument('--batch_state', type=int, default=768)
    parser.add_argument('--batch_retrace', type=int, default=1)
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--update_q_after', type=int, default=100)
    parser.add_argument('--update_pi_after', type=int, default=100)
    parser.add_argument('--iterate_q', type=int, default=15)
    parser.add_argument('--iterate_pi', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    for i in range(args.repeat):
        if args.repeat == 1:
            writer = SummaryWriter('../runs/' + args.name)
        else:
            writer = SummaryWriter('../runs/' + args.name + "_" + str(i))

        mpo_retrace(writer=writer,
                    env=gym.make(args.env),
                    eps_dual=args.eps_dual,
                    eps_mean=args.eps_mean,
                    eps_cov=args.eps_cov,
                    gamma=args.gamma,
                    alpha=args.alpha,
                    sample_episodes=args.sample_episodes,
                    episode_len=args.episode_length,
                    batch_act=args.batch_action,
                    batch_s=args.batch_state,
                    batch_q=args.batch_retrace,
                    update_q_after=args.update_q_after,
                    update_pi_after=args.update_pi_after,
                    update_times_q=args.iterate_q,
                    update_times_pi=args.iterate_pi,
                    epochs=args.epochs
                    )
