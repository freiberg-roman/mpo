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
# parser.add_argument('--epochs', type=int, default=10)

pip install -e
cd eval
python mpo_learn.py \
  --env Pendulum-v0 \
  --name mpo_pendulum_50_50 \
  --epochs 10 \
  --repeat 1 \
  --update_pi_after 50 \
  --update_q_after 50 \
  --iterate_pi 5 \
  --iterate_q 5 \
  --batch_retrace 4 \

# python mpo_learn.py \
#   --env Pendulum-v0 \
#   --name mpo_pendulum_100_100 \
#   --epochs 10 \
#   --repeat 1 \
#   --update_pi_after 100 \
#   --update_q_after 100 \
#   --iterate_pi 5 \
#   --iterate_q 5 \
#   --batch_retrace 4 \

# python mpo_learn.py \
#   --env Pendulum-v0 \
#   --name mpo_pendulum_10_10 \
#   --epochs 10 \
#   --repeat 1 \
#   --update_pi_after 10 \
#   --update_q_after 10 \
#   --iterate_pi 5 \
#   --iterate_q 5 \
#   --batch_retrace 4 \

# python mpo_learn.py \
#   --env Pendulum-v0 \
#   --name mpo_pendulum_1_1_update \
#   --epochs 10 \
#   --repeat 3 \
#   --update_pi_after 1 \
#   --update_q_after 1 \
#   --iterate_pi 5 \
#   --iterate_q 5 \
#   --batch_retrace 2 \

