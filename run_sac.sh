pip install -e
cd eval
python sac.py \
  --env HalfCheetah-v2 \
  --name sac \
  --repeat 1 \

