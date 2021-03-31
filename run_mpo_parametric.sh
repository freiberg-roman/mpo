cd eval
python mpo_retrace_parametric.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_parametric \
  --repeat 1 \
  --total_steps 1000000 \
  --batch_state 128 \
  --min_steps_per_epoch 100 \
  --test_after 5000 \
  --update_steps 100 \
  --update_after 100 \
  --eps_mean 0.005 \
  --eps_cov 0.00001 \
  --rollout_len 5

