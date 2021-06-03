#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 10
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/main_eval.py --run-name checkers_noLP_ppo0504 --run-name-versus checkers_LP_ppo_427  --workers 1 --device cuda:0 --checkpoint checkpoints/eval/%s --stop-after 100000 --num-runs 3  --log runs/checkers/eval/%s  autograph/play/config/checkers/checkers_ensemble_eval.json
