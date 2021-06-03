#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 10
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/main_boardgames_LP.py --run-name chess_noLP_ppo_0409 --workers 1 --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25 --stop-after 1000000 --log runs/chess/ppo/%s  autograph/play/config/chess/chess_noLP_ppo.json
