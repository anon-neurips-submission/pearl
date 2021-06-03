#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 10
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/main_eval.py --run-name chess_mcts --run-name_versus chess_mcts_lp  --workers 1 --device cuda:0 --checkpoint checkpoints/chess/eval/%s --stop-after 100000 --num-runs 25  --log runs/chess/eval/%s  autograph/play/config/chess/chess_ensemble_eval.json