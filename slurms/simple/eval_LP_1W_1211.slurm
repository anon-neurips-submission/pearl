#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 6
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_many_runs.py --run-name LP_1_worker1211 --workers 3 --num-runs 25 --stop-after 100000 --device cuda:0 --checkpoint checkpoints/lp/LPonly_1_1208/%s --do-not-save-checkpoint --log runs/mcts/lp/eval/lp_1W_dfa/%s autograph/play/config/mine_woodfactory/simple_eval_envs/simple_cnn_only.json