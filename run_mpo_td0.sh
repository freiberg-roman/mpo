cd eval
python mpo_td0.py \
  --env Pendulum-v0 \
  --name mpo_td0 \
  --repeat 1 \
  --total_steps 15000 \
  --batch_state 128 \
  --min_steps_per_epoch 100 \
  --test_after 5000 \
  --update_steps 100 \
  --update_after 100 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

