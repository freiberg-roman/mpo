pip install -e
cd eval
python sac_learn.py \
  --env Pendulum-v0 \
  --name sac_pendulum \
  --epochs 1 \
  --repeat 2

python sac_learn.py \
  --env Ant-v2 \
  --name sac_ant \
  --epochs 1 \
  --repeat 2
