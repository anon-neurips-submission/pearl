## PEARL with Boardgames

## boardgames LP 

### To setup environment:

- Use a conda python 3.7 environment
-`conda create -n agrs python=3.7 && conda activate agrs`
- `pip install -r requirements.txt`
- 'Figure out which version of pytorch you need; install from wheel'


YOU SHOULD PROBABLY MODIFY SLURM/SH files so that you can make sure jsons are right.

you first want to train some LP and some noLP games (can be on separate machines). 
Make sure you save checkpoints. You can evaluate loss curves and LP gen curves in log dir runs/
When training is finished, we can use the EVAL script below to run checkpoints against one another. 
The eval will run every checkpoint against each
other so you will have squared the number of checkpoints you have to eval in each folder. 
After eval is finished; put all logs from LP-team in one folder and all logs from noLP team in the other folder. 
'v_LP' is the noLP because it is versus LP and 'versus_noLP' is the LP. 

Then; use the evaluation/getLogs.py script to extract logs from these tensorboard logfiles. 
Then; merge the CSV's you create for each team so you have one large CSV per each team. 

Use the python notebook files to create average curves for each team to compare evaluation. 


# Tips
-Remember to rename run-names based on date so you're not overwriting anything
-create the checkpoints folder if you don't have one.
-Optimized for a serialized, one worker train (A2C), but could experiment with more workers


# CHESS TRAIN SCRIPTS
python autograph/play/main_boardgames_LP.py --run-name chess_LP_mcts_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/mcts/%s  autograph/play/config/chess/chess_LP_mcts.json

python autograph/play/main_boardgames_LP.py --run-name chess_noLP_mcts_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/mcts/%s  autograph/play/config/chess/chess_noLP_mcts.json

python autograph/play/main_boardgames_LP.py --run-name chess_LP_ppo_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/ppo/%s  autograph/play/config/chess/chess_LP_ppo.json

python autograph/play/main_boardgames_LP.py --run-name chess_noLP_ppo_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/ppo/%s  autograph/play/config/chess/chess_noLP_ppo.json

# EVAL two checkpoints CHESS
make sure you put both of the checkpoints in the specified folder. Also; make sure LP-assisted checkpoint is the --run-name_versus


python autograph/play/main_eval.py --run-name chess_mcts --run-name_versus chess_mcts_lp  --workers 1 --device cuda:0 --checkpoint checkpoints/chess/eval/%s
                                        --stop-after 100000 --num-runs 25  --log runs/chess/eval/%s  autograph/play/config/chess/chess_LP_ensemble_eval.json

# GO RUN SCRIPTS

python autograph/play/main_boardgames_LP.py --run-name go_LP_mcts_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/mcts/%s  autograph/play/config/go/go_LP_mcts.json

python autograph/play/main_boardgames_LP.py --run-name go_noLP_mcts_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/mcts/%s  autograph/play/config/go/go_noLP_mcts.json

python autograph/play/main_boardgames_LP.py --run-name go_LP_ppo_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/ppo/%s  autograph/play/config/go/go_LP_ppo.json

python autograph/play/main_boardgames_LP.py --run-name go_noLP_ppo_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/ppo/%s  autograph/play/config/go/go_noLP_ppo.json

# EVAL two checkpoints GO
python autograph/play/main_eval.py --run-name go_mcts --run-name_versus go_mcts_lp  --workers 1 --device cuda:0 --checkpoint checkpoints/go/eval/%s
                                        --stop-after 100000 --num-runs 25  --log runs/go/eval/%s  autograph/play/config/go/go_ensemble_eval.json


# CHECKERS RUN SCRIPTS

python autograph/play/main_boardgames_LP.py --run-name checkers_LP_mcts --workers 1
                                    --device cuda:0 --checkpoint checkpoints/checkers/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/checkers/mcts/%s  autograph/play/config/checkers/checkers_LP_mcts.json

python autograph/play/main_boardgames_LP.py --run-name checkers_noLP_mcts --workers 1
                                    --device cuda:0 --checkpoint checkpoints/checkers/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/checkers/mcts/%s  autograph/play/config/checkers/checkers_noLP_mcts.json


python autograph/play/main_boardgames_LP.py --run-name checkers_LP_ppo --workers 1
                                    --device cuda:0 --checkpoint checkpoints/checkers/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/checkers/ppo/%s  autograph/play/config/checkers/checkers_LP_ppo.json

python autograph/play/main_boardgames_LP.py --run-name checkers_noLP_ppo --workers 1
                                    --device cuda:0 --checkpoint checkpoints/checkers/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/checkers/ppo/%s  autograph/play/config/checkers/checkers_noLP_ppo.json


# EVAL two checkpoints CHECKERS
#!/bin/bash
python autograph/play/main_eval.py --run-name checkers_mcts --run-name_versus checkers_mcts_lp  --workers 1 --device cuda:0 --checkpoint checkpoints/checkers/eval/%s
                                        --stop-after 100000 --num-runs 25  --log runs/checkers/eval/%s  autograph/play/config/checkers/checkers_ensemble_eval.json

