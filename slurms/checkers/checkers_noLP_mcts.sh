#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name checkers_noLP_mcts --workers 1 --device cuda:0 --checkpoint checkpoints/checkers/%s --num-runs 25 --stop-after 1000000 --log runs/checkers/mcts/%s  autograph/play/config/checkers/checkers_noLP_mcts.json
