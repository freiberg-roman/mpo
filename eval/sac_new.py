import argparse
from core.builder.builder_sac import sac
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='Pendulum-v0')
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--eps_dual', type=float, default=0.1)
    parser.add_argument('--eps_mean', type=float, default=0.05)
    parser.add_argument('--eps_cov', type=float, default=0.00001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_action', type=int, default=20)
    parser.add_argument('--batch_state', type=int, default=128)
    parser.add_argument('--rollout_len', type=int, default=5)
    parser.add_argument('--name', type=str, default='sac_new')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--update_steps', type=int, default=50)
    parser.add_argument('--update_after', type=int, default=50)
    parser.add_argument('--total_steps', type=int, default=8000)
    parser.add_argument('--min_steps_per_epoch', type=int, default=50)
    parser.add_argument('--test_after', type=int, default=4000)
    parser.add_argument('--lr_pi', type=float, default=5e-4)
    parser.add_argument('--lr_q', type=float, default=5e-4)
    parser.add_argument('--lr_kl', type=float, default=0.01)
    args = parser.parse_args()

    for i in range(args.repeat):
        if args.repeat == 1:
            writer = SummaryWriter('../runs/' + args.name)
        else:
            writer = SummaryWriter('../runs/' + args.name + "_" + str(i))

        sac(
            env_name=args.env,
            local_device='cuda:0',
            writer=writer,
        )()
