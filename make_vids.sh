#!/usr/bin/bash
source ~/PythonEnv/Torch/bin/activate
MAIN_DIR="/home/raphael/ray_results/DQN/DQN_warehouse_env_e49b5_00000_0_2021-02-03_13-39-44/"
cd $MAIN_DIR
for f in checkpoint_*/; do
  echo $f;
  p=$(echo ${f::-1} | sed -r "s/_/-/g");
  mkdir -p ${f}vids;
  rllib rollout ${f}$p --run DQN --env warehouse_env --out ${f}vids/out.pkl --episodes 5 --video-dir ${f}vids;
done
