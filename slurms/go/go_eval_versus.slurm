#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 10
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/main_eval.py --run-name go_mcts_noLP --run-name-versus go_mcts_LP --workers 1 --device cuda:0 --checkpoint checkpoints/go/%s --num-runs 25 --stop-after 100000 --log runs/go/eval/%s  autograph/play/config/go/go_ensemble_eval.json