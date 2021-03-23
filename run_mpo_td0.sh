cd eval
python mpo_td0.py \
  --env Ant-v2 \
  --name mpo_td0_ant \
  --repeat 1 \
  --total_steps 1000000 \
  --batch_state 128 \
  --min_steps_per_epoch 100 \
  --test_after 5000 \
  --update_steps 100 \
  --update_after 100 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

python mpo_td0.py \
  --env Hopper-v2 \
  --name mpo_td0_h \
  --repeat 1 \
  --total_steps 1000000 \
  --batch_state 128 \
  --min_steps_per_epoch 100 \
  --test_after 5000 \
  --update_steps 100 \
  --update_after 100 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

python mpo_td0.py \
  --env HalfCheetah-v2 \
  --name mpo_td0_ch \
  --repeat 1 \
  --total_steps 1000000 \
  --batch_state 128 \
  --min_steps_per_epoch 100 \
  --test_after 5000 \
  --update_steps 100 \
  --update_after 100 \
  --eps_mean 0.005 \
  --eps_cov 0.00001

