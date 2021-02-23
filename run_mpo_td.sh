pip install -e
cd eval

python mpo_retrace_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_hch_01 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 500 \
  --update_after 250 \
  --eps_mean 0.1 \
  --eps_cov 0.00001 \

python mpo_retrace_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_hch_005 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 200 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 500 \
  --update_after 250 \
  --eps_mean 0.05 \
  --eps_cov 0.00001 \

python mpo_retrace_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_hch_001 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 200 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 500 \
  --update_after 250 \
  --eps_mean 0.01 \
  --eps_cov 0.00001 \

python mpo_retrace_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_hch_01_750 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 200 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 750 \
  --update_after 250 \
  --eps_mean 0.1 \
  --eps_cov 0.00001 \
