pip install -e
cd eval

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1200_100 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1200 \
  --update_after 100

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1500_100 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1500 \
  --update_after 100

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1800_100 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1800 \
  --update_after 100

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1200_300 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1200 \
  --update_after 300

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1500_300 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1500 \
  --update_after 300

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1800_300 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1800 \
  --update_after 300

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_mod_1800_300 \
  --repeat 2 \
  --total_steps 100000 \
  --min_steps_per_epoch 1000 \
  --test_after 4000 \
  --update_steps 1800 \
  --update_after 300 \
  --batch_state 3072
