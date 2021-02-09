pip install -e
cd eval

python mpo_td0_learn_mod.py \
  --env Pendulum-v0 \
  --name mpo_td0_cheetah_sparse_data \
  --total_steps 8000 \
  --repeat 1 \
  --min_steps_per_epoch 200 \
  --test_after 4000

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_cheetah_sparse_data \
  --total_steps 80000 \
  --repeat 2 \
  --min_steps_per_epoch 1000 \
  --test_after 4000
