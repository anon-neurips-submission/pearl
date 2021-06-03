#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 10
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/main_boardgames_LP.py --run-name checkers_noLP_ppo0504 --workers 1 --device cuda:0 --checkpoint checkpoints/checkers/%s --num-runs 10 --stop-after 1000000 --log runs/checkers/ppo/%s  autograph/play/config/checkers/checkers_noLP_ppo.json

