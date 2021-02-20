pip install -e
cd eval

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_hch_250_0005_00001 \
  --repeat 2 \
  --total_steps 60000 \
  --min_steps_per_epoch 100 \
  --test_after 4000 \
  --update_steps 250 \
  --update_after 250 \
  --eps_mean 0.0005 \
  --eps_cov 0.00001 \

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_hch_500_0005_00001 \
  --repeat 2 \
  --total_steps 60000 \
  --min_steps_per_epoch 100 \
  --test_after 4000 \
  --update_steps 500 \
  --update_after 500 \
  --eps_mean 0.0005 \
  --eps_cov 0.00001 \
