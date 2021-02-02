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
python mpo_eta_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_eta_cheetah_10_10 \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 10 \
  --update_q_after 10 \
  --iterate_pi 10 \
  --iterate_q 10 \
  --batch_retrace 8 \
  --episode_length 1000

python mpo_eta_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_eta_cheetah_25_25 \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 25 \
  --update_q_after 25 \
  --iterate_pi 10 \
  --iterate_q 10 \
  --batch_retrace 8 \
  --episode_length 1000

python mpo_eta_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_eta_cheetah_50_50 \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 50 \
  --update_q_after 50 \
  --iterate_pi 10 \
  --iterate_q 10 \
  --batch_retrace 8 \
  --episode_length 1000

python mpo_eta_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_eta_cheetah_100_100 \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 100 \
  --update_q_after 100 \
  --iterate_pi 10 \
  --iterate_q 10 \
  --batch_retrace 8 \
  --episode_length 1000

python mpo_eta_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_eta_cheetah_250_250 \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 250 \
  --update_q_after 250 \
  --iterate_pi 10 \
  --iterate_q 10 \
  --batch_retrace 8 \
  --episode_length 1000

python mpo_eta_learn.py \
  --env HalfCheetah-v2 \
  --name mpo_eta_cheetah_500_500 \
  --epochs 20 \
  --repeat 3 \
  --update_pi_after 500 \
  --update_q_after 500 \
  --iterate_pi 10 \
  --iterate_q 10 \
  --batch_retrace 8 \
  --episode_length 1000

