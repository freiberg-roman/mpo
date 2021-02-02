pip install -e
cd eval
python sac_learn.py \
  --env Pendulum-v0 \
  --name sac_learn \
  --epochs 10 \
  --repeat 3

python sac_learn.py \
  --env Ant-v2 \
  --name sac_learn \
  --epochs 10 \
  --repeat 3
