import math
import os
import signal
import sys
from typing import Callable, Any, Tuple, List, Union, Optional

import ptan
import torch
import numpy
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import multiprocessing
from torch.optim import Adam, SGD

import autograph.lib.envs.mazeenv
from autograph.lib.automata import AutomatonSet
from autograph.lib.envs.mazeenv import FuelMazeEnv, FuelMazeObservation
from autograph.lib.envs.mazeenv import transform_coordinate
from autograph.lib.envs.mineworldenv import MineWorldEnv
from autograph.lib.loss_functions import TakeSimilarActionsLossFunction, PPOLossFunction, \
    AdvantageActorCriticLossFunction
from autograph.lib.envs.go.go_env import env as goEnvironment
from autograph.lib.envs.chess.chess_env import env as chessEnv
from autograph.lib.envs.checkers import env as checkersEnv

from autograph.lib.running_ensemble_chess import get_parallel_queue as get_parallel_queue_chess
from autograph.lib.running_ensemble_chess import EvalLoop as EvalLoop_chess
from autograph.lib.running_ensemble_all import get_parallel_queue, EvalLoop, \
    run_episode_generic_eval_chess, run_episode_generic_eval_go, run_episode_generic_eval_checkers

from autograph.lib.shaping import AutShapingWrapper
from autograph.lib.util import element_add
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler, PickleLoadHandler, TransplantCheckpointManager
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer, NoopCuriosityOptimizer
from autograph.net.maze_constructors import mazenet_v1, mazernd_v1, maze_obs_rewrite_creator
from autograph.net.mine_constructors import minenet_v1, mine_obs_rewriter_creator, minernd_v1, \
    mine_mazenet_v1, mine_mazenet_v1_go, mine_obs_go_rewriter_creator, mine_obs_chess_rewriter_creator, \
    mine_mazenet_v1_chess, mine_mazenet_v1_checkers, mine_obs_checkers_rewriter_creator
from autograph.net.misc_constructors import gym_make, no_op_cur_make, basic_net, no_op_make
from lp.LP_general_goEnv import get_confusing_input_pool
math.sqrt(1)  # So that the import isn't optimized away (very useful when setting conditional debug breakpoints)

sys.modules["autograph.lib.mazeenv"] = autograph.lib.envs.mazeenv  # Fix broken pickle loading


def throwKeyInterr():
    raise KeyboardInterrupt()


optimizers = {
    "Adam": Adam,
    "SGD": SGD
}

def no_op_rewriter(x):
    return torch.Tensor([0.0])


training_nets = {
    "mine_mazenet_v1_go": (mine_mazenet_v1_go, mine_obs_go_rewriter_creator),
    "mine_mazenet_v1_chess": (mine_mazenet_v1_chess, mine_obs_chess_rewriter_creator),
    "mine_mazenet_v1_checkers": (mine_mazenet_v1_checkers, mine_obs_checkers_rewriter_creator),
    "mine_mazenet_v1": (mine_mazenet_v1, mine_obs_rewriter_creator),
    "basicnet": (basic_net, lambda e: torch.Tensor),
    "no-op": (no_op_make, lambda e: no_op_rewriter)
}

curiosity_nets = {
    "mazernd_v1": (mazernd_v1, maze_obs_rewrite_creator),
    "minernd_v1": (minernd_v1, mine_obs_rewriter_creator),
    "no-op": (no_op_cur_make, no_op_rewriter)
}

loss_funcs = {
    "MCTS": TakeSimilarActionsLossFunction,
    "PPO": PPOLossFunction,
    "A2C": AdvantageActorCriticLossFunction
}


if __name__ == '__main__':
    import argparse
    import json5 as json

    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--log")
    p.add_argument("--checkpoint")
    p.add_argument("--checkpoint-versus")

    p.add_argument("--run-name")
    p.add_argument("--run-name-versus")

    p.add_argument("--do-not-load-from-checkpoint", dest="load_checkpoint", action="store_true")
    p.add_argument("--do-not-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--checkpoint-every", default=1)
    p.add_argument("--workers", default=1)
    p.add_argument("--post", help="Add a postfix to the checkpoint and tensorboard names")


    # WE ARE EDITING THIS FOR THE BULK STATIC ENV RUNS
    p.add_argument("--num-runs", dest="num_runs", default=100)
    p.add_argument("--stop-after", dest="stop_after", default=3000000,
                   help="Stop after roughly a certain number of steps have been reached")
    args = vars(p.parse_args())



    run_name = args.get("run_name")
    STOP_AFTER = args.get("stop_after")
    NUM_RUNS = args.get("num_runs")

    if STOP_AFTER:
        STOP_AFTER = int(STOP_AFTER)

    if NUM_RUNS:
        NUM_RUNS = int(NUM_RUNS)

    def interpolate(text):
        if not text:
            return text

        if run_name and "%s" in text:
            return text % (run_name,)
        else:
            return text


    config_file = interpolate(args["config"])

    postfix = ""

    if args.get("post"):
        postfix = "_" + args["post"]

    with open(config_file) as f:
        config = json.load(f)



    DISCOUNT = config["discount"]


    env = config["env"]
    MAX_EPISODE_LEN = env.get("max_episode_len", 500)
    MAX_LEN_REWARD = env.get("max_len_reward", 0)
    ENV_CONFIG = env.get("params", None)
    ENV_TYPE = env.get("type")
    BOARD_SIZE = env.get("board_size", 7)
    # Policy training hyperparameters
    training: dict = config["training"]

    LEARNING_RATE = training["learning_rate"]
    REPLAY_BUFFER = training["replay_buffer"]
    MIN_TRACE_TO_TRAIN = training["min_trace_to_train"]
    PPO_TRAIN_ROUNDS = training["train_rounds"]
    NETWORK = training.get("network", "mazenet_v1")
    NETWORK_PARAMS = training.get("params", dict())

    OPTIMIZER = optimizers[training.get("optimizer")]
    OPTIMIZER_PARAMS = training.get("opt_params", {})

    ALL_WORK_LP = training.get("all_work_lp", False)

    # Loss function
    loss: dict = config.get("loss")
    if loss:
        LOSS_FUNC = loss["type"]
        LOSS_PARAMS = loss.get("params", dict())
    else:
        # changed from "MCTS" when adding deepQ no MCTS ensemble config
        LOSS_FUNC = "PPO"
        LOSS_PARAMS = dict()

    if config.get("mcts"):
        config["episode_runner"] = {
            "type": "mcts_aut_episode",
            "params": config.pop("mcts")
        }

    # Policy runner parameters
    episode_runner = config["episode_runner"]
    EPISODE_RUNNER_TYPE = episode_runner["type"]
    EPISODE_RUNNER_PARAMS = episode_runner.get("params", dict())


    # Logging and checkpointing

    LOG_FOLDER = interpolate(args.get("log")) + postfix

    CHECKPOINT_EVERY = int(args["checkpoint_every"])
    CHECKPOINT_PATH = interpolate(args.get("checkpoint")) + postfix
    CHECKPOINT_PATH_versus = interpolate(args.get("checkpoint-versus", CHECKPOINT_PATH)) + postfix


    if CHECKPOINT_PATH:
        LOAD_FROM_CHECKPOINT = args["load_checkpoint"]
        if not os.path.isfile(CHECKPOINT_PATH):
            LOAD_FROM_CHECKPOINT = True
            print("NOTE: no existing checkpoint found, will create new one if checkpoint saving is enabled.")


        SAVE_CHECKPOINTS = args["save_checkpoint"]

    NUM_PROCESSES = int(args["workers"])
    DEVICE = torch.device(args["device"])


def run_eval_ensemble_go(nets: List[torch.nn.Module], env: AutShapingWrapper, max_length: int,
                         max_len_reward: Union[int, None],
                         device, train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None, lp_gen=False,
                         jensen_shannon=None, output_value=None, **kwargs) \
        -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """



    # uses action probs from net with the higher expected value from the net
    def action_value_generator(state, turn):

        net_usual = 0 if turn == 0 else 2
        # the mcts (noLP) player doesn't use his 2nd net like our LP does.
        net_patho = 0 if turn == 0 else 3


        states = [state]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[net_usual](states_transformed.to(device))
        pols1, vals1 = nets[net_patho](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        if val1 > val:
            val = val1
            pollist = pollist1

        return pollist, val


    return run_episode_generic_eval_go(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=False, jensen_shannon=jensen_shannon, output_valueS=output_value)

def run_chess_eval(nets: List[torch.nn.Module], env: AutShapingWrapper, max_length: int,
                         max_len_reward: Union[int, None],
                         device, train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None,lp_gen=False,
                         jensen_shannon=None, output_value=None, **kwargs) \
        -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    # uses action probs from net with the higher expected value from the net
    def action_value_generator(state, turn):
        net_usual = 0 if turn == 0 else 2
        # the mcts (noLP) player doesn't use his 2nd net like our LP does.
        net_patho = 0 if turn == 0 else 3

        states = [state[0]]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[net_usual](states_transformed.to(device))
        pols1, vals1 = nets[net_patho](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        if val1 > val:
            val = val1
            pollist = pollist1

        return pollist, val


    return run_episode_generic_eval_chess(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=False, jensen_shannon=jensen_shannon, output_valueS=output_value)

def run_checkers_eval(nets: List[torch.nn.Module], env: AutShapingWrapper, max_length: int,
                         max_len_reward: Union[int, None],
                         device, train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None,lp_gen=False,
                         jensen_shannon=None, output_value=None, **kwargs) \
        -> Tuple[List[TraceStep], float]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    # uses action probs from net with the higher expected value from the net
    def action_value_generator(state, turn):
        net_usual = 0 if turn == 0 else 2
        # the mcts (noLP) player doesn't use his 2nd net like our LP does.
        net_patho = 0 if turn == 0 else 3
        observation = state['observation']
        states = [observation]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[net_usual](states_transformed.to(device))
        pols1, vals1 = nets[net_patho](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        if val1 > val:
            val = val1
            pollist = pollist1

        return pollist, val


    return run_episode_generic_eval_checkers(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=False, jensen_shannon=jensen_shannon, output_valueS=output_value)

episode_runners = {
    "ensemble_eval_versus_go": run_eval_ensemble_go,
    "ensemble_eval_versus_chess": run_chess_eval,
    "ensemble_eval_versus_checkers": run_checkers_eval,

}


def run():
    torch.multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGHUP, throwKeyInterr)

    LOAD_FROM_CHECKPOINT = True


    # all against each other up to NUM_RUNS so if 8, 64 - 8*2 games.
    for tr_idx in range(0, NUM_RUNS):
        for other_idx in range(0, NUM_RUNS):

            #RUN_POSTFIX = "_tr0_copy"
            RUN_POSTFIX = "_tr" + str(tr_idx) + '_copy'
            RUN_POSTFIX_OPP = "_tr" + str(other_idx) + '_copy'
            #RUN_POSTFIX2 = "_tr" + str(1) + "_copy"

            #RUN_POSTFIX = ""
            try:
                cman = CheckpointManager(CHECKPOINT_PATH + RUN_POSTFIX, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
            except EOFError:
                cman = CheckpointManager(CHECKPOINT_PATH + RUN_POSTFIX + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)

            try:
                cman_versus = CheckpointManager(CHECKPOINT_PATH_versus + RUN_POSTFIX_OPP, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
            except EOFError:
                cman_versus = CheckpointManager(CHECKPOINT_PATH_versus + RUN_POSTFIX_OPP + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)

            if EPISODE_RUNNER_TYPE == "ensemble_eval_versus_go":
                # initialize go environment.
                env = goEnvironment(board_size=BOARD_SIZE, board=None, komi=4)
            elif EPISODE_RUNNER_TYPE == "ensemble_eval_versus_chess":
                env = chessEnv()
            else:
                env = checkersEnv()

            writer = SummaryWriter(LOG_FOLDER + RUN_POSTFIX +'v_LP_' + RUN_POSTFIX_OPP)
            writer_versus = SummaryWriter(LOG_FOLDER + RUN_POSTFIX_OPP + 'versus_noLP' + RUN_POSTFIX)

            train_net_creator, train_rewriter_creator = training_nets[NETWORK]

            # load both nets for player 1
            net = cman.load("net", train_net_creator(env, **NETWORK_PARAMS),
                            CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

            net_lp = cman.load("net_lp", train_net_creator(env, **NETWORK_PARAMS),
                               CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)


            # load both nets for player 2
            net_versus = cman_versus.load("net", train_net_creator(env, **NETWORK_PARAMS),
                            CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

            net_lp_versus = cman_versus.load("net_lp", train_net_creator(env, **NETWORK_PARAMS),
                               CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

            net.share_memory()
            net_lp.share_memory()

            net_versus.share_memory()
            net_lp_versus.share_memory()

            nets = [net, net_lp, net_versus, net_lp_versus]
            writers = [writer, writer_versus]
            loss_functions = [loss_funcs[LOSS_FUNC](net=nets[0], device=DEVICE, discount=DISCOUNT, **LOSS_PARAMS),
                              loss_funcs[LOSS_FUNC](net=nets[1], device=DEVICE, discount=DISCOUNT, **LOSS_PARAMS)]

            optimizer = cman.load("opt", OPTIMIZER(net.parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                                  StateDictLoadHandler())
            optimizer_lp = cman.load("opt_lp", OPTIMIZER(net_lp.parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                                     StateDictLoadHandler())
            optimizers = [optimizer, optimizer_lp]


            train_rewriter = train_rewriter_creator(env)

            if EPISODE_RUNNER_TYPE != "ensemble_eval_versus_chess":
                train_loop = EvalLoop(writers, DEVICE)
            else:
                train_loop = EvalLoop_chess(writers, DEVICE)
            # train rounds 0 because eval
            #
            # TODO test checkers EVAL


            with get_parallel_queue(num_processes=NUM_PROCESSES, episode_runner=episode_runners[EPISODE_RUNNER_TYPE],
                                    nets=nets, env=env, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
                                    state_observer=None, device=DEVICE,
                                    train_state_rewriter=train_rewriter,
                                    lp_gen=False, config_file=config_file, all_work_lp=ALL_WORK_LP,
                                    **EPISODE_RUNNER_PARAMS) as sim_round_queue:
                while True:
                    train_loop(sim_round_queue, loss_functions, optimizers)

                    if train_loop.num_rounds % CHECKPOINT_EVERY == 0:

                        if STOP_AFTER and train_loop.global_step > STOP_AFTER:
                            print("STOPPING: step limit " + str(train_loop.global_step) + "/" + str(STOP_AFTER))

                            break




if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
