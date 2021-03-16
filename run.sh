pip install -e
cd eval
python mpo_retrace_parametric.py \
  --env Pendulum-v0 \
  --name mpo_retrace_200_01_10 \
  --repeat 1 \
  --total_steps 12000 \
  --batch_state 128 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 200 \
  --update_after 200 \
  --eps_mean 0.01 \
  --eps_cov 0.00001 \
  --rollout_len 10

