pip install -e
cd eval

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_q_polyak_sac_update_50 \
  --repeat 3 \
  --total_steps 160000 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 50 \
  --update_after 50

python mpo_td0_learn_mod_sac_update.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_q_polyak_sac_update_100 \
  --repeat 3 \
  --total_steps 160000 \
  --min_steps_per_epoch 50 \
  --test_after 4000 \
  --update_steps 100 \
  --update_after 100
