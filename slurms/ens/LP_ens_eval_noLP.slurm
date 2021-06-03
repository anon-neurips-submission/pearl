#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 4
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_double_eval.py --run-name noLP_pool_mpi3_ --workers 3 --device cuda:0 --checkpoint checkpoints/lp/lp_ens_221_copy/%s --do-not-save-checkpoint --num-runs 25 --stop-after 5060000 --log runs/ppo/noLP/%s  autograph/play/config/mine_woodfactory/simple_eval_envs/simple_LP_ensemble_eval.json
