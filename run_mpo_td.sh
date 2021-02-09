pip install -e
cd eval

python mpo_td0_learn_mod.py \
  --env Pendulum-v0 \
  --name mpo_pendulum_td0_low_data_ \
  --total_steps 12000 \
  --repeat 3 \
  --min_steps_per_epoch 200 \
  --update_steps 1200 \
  --update_after 100 \
  --test_after 1000

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_cheetah_td0_low_data_1200_100 \
  --total_steps 160000 \
  --repeat 3 \
  --min_steps_per_epoch 1000 \
  --update_steps 1200 \
  --update_after 100 \
  --test_after 4000

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_cheetah_td0_low_data_1200_200 \
  --total_steps 160000 \
  --repeat 3 \
  --min_steps_per_epoch 1000 \
  --update_steps 1200 \
  --update_after 200 \
  --test_after 4000

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_cheetah_td0_low_data_1200_300 \
  --total_steps 160000 \
  --repeat 3 \
  --min_steps_per_epoch 1000 \
  --update_steps 1200 \
  --update_after 300 \
  --test_after 4000

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_cheetah_td0_low_data_1500_100 \
  --total_steps 160000 \
  --repeat 3 \
  --min_steps_per_epoch 1000 \
  --update_steps 1500 \
  --update_after 100 \
  --test_after 4000

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_cheetah_td0_low_data_2000_100 \
  --total_steps 160000 \
  --repeat 3 \
  --min_steps_per_epoch 1000 \
  --update_steps 2000 \
  --update_after 100 \
  --test_after 4000
