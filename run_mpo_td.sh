pip install -e
cd eval

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_2400_100_long \
  --repeat 2 \
  --total_steps 300000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 2400 \
  --update_after 100

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1800_100_long \
  --repeat 2 \
  --total_steps 300000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1800 \
  --update_after 100
