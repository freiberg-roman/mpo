pip install -e
cd eval

python mpo_retrace_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_hch_1200_300 \
  --repeat 3 \
  --total_steps 60000 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 1200 \
  --update_after 300 \
  --eps_mean 0.0005 \
  --eps_cov 0.00001 \

python mpo_retrace_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_retrace_hch_1500_300 \
  --repeat 3 \
  --total_steps 60000 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 1500 \
  --update_after 300 \
  --eps_mean 0.0005 \
  --eps_cov 0.00001 \

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_hch_1200_300 \
  --repeat 3 \
  --total_steps 60000 \
  --min_steps_per_epoch 200 \
  --test_after 4000 \
  --update_steps 1200 \
  --update_after 300 \
  --eps_mean 0.0005 \
  --eps_cov 0.00001 \

