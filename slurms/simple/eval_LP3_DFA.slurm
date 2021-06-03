#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 6
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_many_runs.py --run-name LP_20_3_worker1 --workers 3 --num-runs 25 --stop-after 100000 --device cuda:0 --checkpoint checkpoints/lp/LP3_20_1LP_DFA1/%s --do-not-save-checkpoint --log runs/mcts/lp/eval/lp3_DFA/%s autograph/play/config/mine_woodfactory/simple_eval_envs/simple_cnn_only.json