import argparse
from core.sac.sac import sac
import gym
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--name', type=str, default='sac')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    for i in range(args.repeat):
        if args.repeat == 1:
            writer = SummaryWriter('../runs/' + args.name)
        else:
            writer = SummaryWriter('../runs/' + args.name + "_" + str(i))

        sac(env_fn=lambda: gym.make(args.env),
            writer=writer,
            epochs=args.epochs)
