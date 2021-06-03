#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 10
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/main_boardgames_LP.py --run-name go_LP_ppo_0409 --workers 1 --device cuda:0 --checkpoint checkpoints/go/%s --num-runs 25 --stop-after 1000000 --log runs/go/ppo/%s  autograph/play/config/go/go_LP_ppo.json