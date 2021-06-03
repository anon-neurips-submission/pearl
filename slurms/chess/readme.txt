YOU SHOULD PROBABLY MAKE YOUR OWN SLURM/SH files so that you can make sure jsons are right.

TODO: All the main_*.py files are merged into main_boardgames_LP.py

so make this the first file in your runfile


# rename run-names based on date so you're not overwriting anything

# CHESS TRAIN SCRIPTS

#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name chess_LP_mcts_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/mcts/%s  autograph/play/config/chess/chess_LP_mcts.json
#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name chess_noLP_mcts_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/mcts/%s  autograph/play/config/chess/chess_noLP_mcts.json

#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name chess_LP_ppo_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/ppo/%s  autograph/play/config/chess/chess_LP_ppo.json
#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name chess_noLP_ppo_0409 --workers 1
                                        --device cuda:0 --checkpoint checkpoints/chess/%s --num-runs 25
                                        --stop-after 1000000 --log runs/chess/ppo/%s  autograph/play/config/chess/chess_noLP_ppo.json

# EVAL two checkpoints CHESS
make sure you put both of the checkpoints in the specified folder. Also; make sure LP-assisted checkpoint is the --run-name_versus

#!/bin/bash
python autograph/play/main_eval.py --run-name chess_mcts --run-name_versus chess_mcts_lp  --workers 1 --device cuda:0 --checkpoint checkpoints/chess/eval/%s
                                        --stop-after 100000 --num-runs 25  --log runs/chess/eval/%s  autograph/play/config/chess/chess_LP_ensemble_eval.json


# GO RUN SCRIPTS

#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name go_LP_mcts_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/mcts/%s  autograph/play/config/go/go_LP_mcts.json

#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name go_noLP_mcts_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/mcts/%s  autograph/play/config/go/go_noLP_mcts.json

#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name go_LP_ppo_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/ppo/%s  autograph/play/config/go/go_LP_ppo.json

#!/bin/bash
python autograph/play/main_boardgames_LP.py --run-name go_noLP_ppo_0409 --workers 1
                                    --device cuda:0 --checkpoint checkpoints/go/%s
                                    --num-runs 25 --stop-after 1000000 --log runs/go/ppo/%s  autograph/play/config/go/go_noLP_ppo.json

# EVAL two checkpoints GO
#!/bin/bash
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
