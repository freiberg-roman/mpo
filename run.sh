pip install -e
cd eval

python mpo_retrace.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_005_00001 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 50 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

python mpo_retrace_parametric.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_parametric_005_00001 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 50 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

python mpo_td0.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_005_00001 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 50 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

python mpo_td0_parametric.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_parametric_005_00001 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 50 \
  --eps_mean 0.005 \
  --eps_cov 0.00001
