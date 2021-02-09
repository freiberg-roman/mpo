pip install -e
cd eval

python mpo_td0_learn_mod.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_cheetah_sparse_data \
  --epochs 80 \
  --repeat 2 \
  --episode_length 1000 \
  --min_steps_per_epoch 1000 \
