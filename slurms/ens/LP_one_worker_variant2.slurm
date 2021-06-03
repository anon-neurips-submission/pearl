#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 16
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_double.py --run-name LP_ens_variant2 --workers 1 --device cuda:0 --checkpoint checkpoints/lp/lp_ens/%s --num-runs 25 --stop-after 60000 --log runs/mcts/lp/lp_ens_v2/%s  autograph/play/config/mine_woodfactory/simple_eval_envs/simple_LP_ensemble.json
