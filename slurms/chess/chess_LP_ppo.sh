#!/bin/bash

python autograph/play/main_boardgames_LP.py --run-name chess_LP_ppo_0409 --workers 1 --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25 --stop-after 1000000 --log runs/chess/ppo/%s  autograph/play/config/chess/chess_LP_ppo.json
