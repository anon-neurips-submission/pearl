#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name chess_noLP_mcts_0409 --workers 1 --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25 --stop-after 1000000 --log runs/chess/mcts/%s  autograph/play/config/chess/chess_noLP_mcts.json
