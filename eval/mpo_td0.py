import argparse
from core.builder.builder_mpo import mpo_non_parametric_td0
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
    parser.add_argument('--name', type=str, default='debug-2')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--update_steps', type=int, default=100)
    parser.add_argument('--update_after', type=int, default=100)
    parser.add_argument('--total_steps', type=int, default=12000)
    parser.add_argument('--min_steps_per_epoch', type=int, default=100)
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

        mpo_non_parametric_td0(
            env_name=args.env,
            local_device='cpu',
            writer=writer,
            lr_pi=args.lr_pi,
            lr_q=args.lr_q,
            lr_kl=args.lr_kl,
            eps_dual=args.eps_dual,
            eps_mean=args.eps_mean,
            eps_cov=args.eps_cov,
            gamma=args.gamma,
            batch_size=args.batch_state,
            batch_size_act=args.batch_action,
            update_steps=args.update_steps,
            update_after=args.update_after,
            min_steps_per_epoch=args.min_steps_per_epoch,
            test_after=args.test_after,
            total_steps=args.total_steps
        )()
