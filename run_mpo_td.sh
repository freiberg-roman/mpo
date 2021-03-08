pip install -e
cd eval

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_hch_005 \
  --repeat 2 \
  --total_steps 60000 \
  --batch_state 128 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 250 \
  --eps_mean 0.005 \
  --eps_cov 0.00001 \
