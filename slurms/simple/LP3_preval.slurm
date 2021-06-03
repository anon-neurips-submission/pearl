#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 6
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_many_runs.py --run-name LP3prev/LP_3_worker --workers 3 --device cuda:0 --checkpoint checkpoints/lp/%s  --num-runs 25 --stop-after 80000 --log runs/mcts/lp/%s  autograph/play/config/mine_woodfactory/simple_eval_envs/simple_combined_LP.json
