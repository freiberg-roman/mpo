# parser.add_argument('--env', type=str, default='Pendulum-v0')
# parser.add_argument('--eps_dual', type=float, default=0.1)
# parser.add_argument('--eps_mean', type=float, default=0.1)
# parser.add_argument('--eps_cov', type=float, default=0.0001)
# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--alpha', type=float, default=10.)
# parser.add_argument('--sample_episodes', type=int, default=1)
# parser.add_argument('--episode_length', type=int, default=200)
# parser.add_argument('--batch_action', type=int, default=20)
# parser.add_argument('--batch_state', type=int, default=768)
# parser.add_argument('--batch_retrace', type=int, default=1)
# parser.add_argument('--name', type=str, default='debug')
# parser.add_argument('--repeat', type=int, default=1)
# parser.add_argument('--update_q_after', type=int, default=100)
# parser.add_argument('--update_pi_after', type=int, default=100)
# parser.add_argument('--iterate_q', type=int, default=15)
# parser.add_argument('--iterate_pi', type=int, default=5)
# parser.add_argument('--epochs', type=int, default=9)
# parser.add_argument('--lr_pi', type=float, default=5e-4
# parser.add_argument('--lr_q', type=float, default=2e-4)

pip install -e
cd eval

python mpo_td0_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_cheetha_entropy_adam_lagrauge \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 20 \
  --update_q_after 20 \
  --episode_length 1000 \
  --q_alpha 0.2

python mpo_td0_learn.py \
  --env HalfCheetah-v2 \
  --name epo_td0_cheetha_adam_lagrauge \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 20 \
  --update_q_after 20 \
  --episode_length 1000 \
  --q_alpha 0.0
