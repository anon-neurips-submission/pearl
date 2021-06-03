echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python autograph/play/maze_nn_aut_LP_many_runs.py --run-name LP3_20_1LP_DFA/LP_20_3_worker --workers 3 --device cuda:2 --checkpoint checkpoints/lp/%s --num-runs 25 --stop-after 60000 --log runs/mcts/lp/%s  autograph/play/config/mine_woodfactory/simple_eval_envs/simple_combined_LP.json
