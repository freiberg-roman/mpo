pip install -e
cd eval

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_q_polyak_3600_100 \
  --repeat 2 \
  --total_steps 160000 \
  --min_steps_per_epoch 4000 \
  --test_after 4000 \
  --update_steps 3600 \
  --update_after 100

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_q_polyak_5000_100 \
  --repeat 2 \
  --total_steps 160000 \
  --min_steps_per_epoch 4000 \
  --test_after 4000 \
  --update_steps 5000 \
  --update_after 100

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_q_polyak_3600_50 \
  --repeat 2 \
  --total_steps 160000 \
  --min_steps_per_epoch 4000 \
  --test_after 4000 \
  --update_steps 5000 \
  --update_after 50
