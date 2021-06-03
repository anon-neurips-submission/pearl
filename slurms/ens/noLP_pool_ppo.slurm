#!/bin/bash
#SBATCH --gres=gpu:pascal:1
#SBATCH -c 8
#SBATCH -p external
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_double.py --run-name noLP_pool_mpi3_ --workers 3 --device cuda:0 --checkpoint checkpoints/lp/lp_ens/%s --checkpoint-every 5 --num-runs 25 --stop-after 5000000 --log runs/ppo/lp/%s  autograph/play/config/mine_woodfactory/simple_eval_envs/simple_noLP_easy.json
