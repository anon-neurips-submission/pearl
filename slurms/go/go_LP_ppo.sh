python autograph/play/main_boardgames_LP.py --run-name go_LP_ppo_0409 --workers 1 --device cuda:0 --checkpoint checkpoints/go/%s --num-runs 25 --stop-after 1000000 --log runs/go/ppo/%s  autograph/play/config/go/go_LP_ppo.json