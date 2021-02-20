pip install -e
cd eval
# parser.add_argument('--eps_mean', type=float, default=0.001)
# parser.add_argument('--eps_cov', type=float, default=0.000001)

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_q_polyak_sac_update_50 \
  --repeat 3 \
  --total_steps 160000 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 50 \
  --eps_mean 0.001 \
  --eps_cov 0.00001 \

