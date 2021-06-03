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

from autograph.lib.loss_functions import TakeSimilarActionsLossFunction, PPOLossFunction, \
    AdvantageActorCriticLossFunction
from autograph.lib.envs.go.go_env import env as goEnvironment
from autograph.lib.envs.checkers import env as checkersEnv
from autograph.lib.envs.chess.chess_env import env as chessEnv

from autograph.lib.mcts_go import MCTS as MCTS_GO
from autograph.lib.mcts_checkers import MCTS as MCTS_CHECKERS
from autograph.lib.mcts_chess import MCTS as MCTS_CHESS

from autograph.lib.running_ensemble_all import get_parallel_queue, RandomReplayTrainingLoop, \
    run_episode_generic_lp_go, run_episode_generic_lp_chess, run_episode_generic_lp_checkers
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler, PickleLoadHandler, TransplantCheckpointManager
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.maze_constructors import mazenet_v1, mazernd_v1, maze_obs_rewrite_creator
from autograph.net.mine_constructors import minenet_v1, mine_obs_rewriter_creator, minernd_v1, \
    mine_mazenet_v1, mine_mazenet_v1_go, mine_obs_go_rewriter_creator, \
    mine_mazenet_v1_chess, mine_obs_chess_rewriter_creator, mine_mazenet_v1_checkers, mine_obs_checkers_rewriter_creator
from autograph.net.misc_constructors import gym_make, no_op_cur_make, basic_net, no_op_make
from autograph.play.main_chess_LP import throwKeyInterr

DEBUG = False


math.sqrt(1)  # So that the import isn't optimized away (very useful when setting conditional debug breakpoints)

sys.modules["autograph.lib.mazeenv"] = autograph.lib.envs.mazeenv  # Fix broken pickle loading

optimizers = {
    "Adam": Adam,
    "SGD": SGD
}

env_constructors = {
    "go": goEnvironment,
    "chess": chessEnv,
    "checkers": checkersEnv
}

def no_op_rewriter(x):
    return torch.Tensor([0.0])


# We will duplicate the architecture of these nets for LP pathological nets
training_nets = {
    "mine_mazenet_v1_go": (mine_mazenet_v1_go, mine_obs_go_rewriter_creator),
    "mine_mazenet_v1_chess": (mine_mazenet_v1_chess, mine_obs_chess_rewriter_creator),
    "mine_mazenet_v1_checkers": (mine_mazenet_v1_checkers, mine_obs_checkers_rewriter_creator),

    "mine_mazenet_v1": (mine_mazenet_v1, mine_obs_rewriter_creator),
    "basicnet": (basic_net, lambda e: torch.Tensor),
    "no-op": (no_op_make, lambda e: no_op_rewriter)
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
    p.add_argument("--run-name")
    p.add_argument("--do-not-load-from-checkpoint", dest="load_checkpoint", action="store_false")
    p.add_argument("--do-not-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--checkpoint-every", default=1)
    p.add_argument("--workers", default=8)
    p.add_argument("--post", help="Add a postfix to the checkpoint and tensorboard names")


    # WE ARE EDITING THIS FOR THE BULK STATIC ENV RUNS
    p.add_argument("--num-runs", dest="num_runs", default=100)
    p.add_argument("--stop-after", dest="stop_after", default=30000,
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
    # DEFAULT is go, specify type  in config to get chess or checkers
    ENV_TYPE = env.get("type", "go")
    LP_EVERY_X_STATES = env.get("lp_every_x_states", 3)
    LP_POOL_SIZE = env.get("lp_pool_size", 5)


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


    if EPISODE_RUNNER_TYPE == "go_mcts_LP" or EPISODE_RUNNER_TYPE == "go_mcts_noLP" or \
            EPISODE_RUNNER_TYPE == "checkers_mcts_LP" or EPISODE_RUNNER_TYPE == "checkers_mcts" or \
            EPISODE_RUNNER_TYPE == "chess_mcts_LP" or EPISODE_RUNNER_TYPE == "chess_mcts_noLP":
        LOSS_FUNC = "MCTS"

    # Logging and checkpointing

    LOG_FOLDER = interpolate(args.get("log")) + postfix
    CHECKPOINT_EVERY = int(args["checkpoint_every"])
    CHECKPOINT_PATH = interpolate(args.get("checkpoint")) + postfix


    if CHECKPOINT_PATH:
        LOAD_FROM_CHECKPOINT = args["load_checkpoint"]
        if not os.path.isfile(CHECKPOINT_PATH):
            LOAD_FROM_CHECKPOINT = False
            print("NOTE: no existing checkpoint found, will create new one if checkpoint saving is enabled.")

        SAVE_CHECKPOINTS = args["save_checkpoint"]
    else:
        CHECKPOINT_PATH = None
        LOAD_FROM_CHECKPOINT = False
        SAVE_CHECKPOINTS = False
        if not args["save_checkpoint"]:
            print("WARNING: This run is not being checkpointed! Use --do-not-save-checkpoint to suppress.")


    # TODO: optimize LP for more than 1 workers. should scale fine but most research has been done with 1 worker
    NUM_PROCESSES = int(args["workers"])
    DEVICE = torch.device(args["device"])


def run_go_mcts_ensemble(nets: List[torch.nn.Module], env: goEnvironment, max_length: int,
                         max_len_reward: Union[int, None],
                         device, train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None,lp_gen=False,
                         jensen_shannon=None, output_value=None, num_batches=4, batch_size=5, **kwargs) \
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

    # We will do the action masking here; so the MCTS is only doing legal actions.
    def state_evaluator(states, lp_genn=False):
        if lp_genn:
            index_of_net = 1
        else:
            index_of_net = 0
        # index_of_net = 0
        states = [state['observation'] for state in states]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[index_of_net](states_transformed.to(device))
        valss = torch.tanh(vals)
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = valss.squeeze(-1).tolist()

        return list(zip(pollist, vallist))


    mcts = MCTS_GO(env.env.env.env.env.action_space.n, None, state_evaluator, None, **kwargs)

    def action_value_generator(state, lp_genn):
        mcts.mcts_batch(env, state, num_batches, batch_size, lp_genn)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)

    return run_episode_generic_lp_go(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)





def run_go_ppo(nets: List[torch.nn.Module], env: goEnvironment, max_length: int,
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


    def action_value_generator(state, net_to_use):
        observation = state['observation']
        states = [observation]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))

        pols, vals = nets[0](states_transformed.to(device))
        pols1, vals1 = nets[1](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        # hard code
        if not lp_gen:
            net_to_use = 0

        if net_to_use == 1:
            val = val1
            pollist = pollist1
        elif net_to_use > 1:
            # use the net that gives us the highest value from our current state.
            if val1 > val:
                val = val1
                pollist = pollist1

        return pollist, val

    return run_episode_generic_lp_go(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)


def run_chess_mcts_ensemble(nets: List[torch.nn.Module], env: chessEnv, max_length: int,
                         max_len_reward: Union[int, None],
                         device, train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None,lp_gen=True,
                         jensen_shannon=None, output_value=None, num_batches=4, batch_size=5, **kwargs) \
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

    # We will do the action masking here; so the MCTS is only doing legal actions.
    def state_evaluator(states, lp_genn=False):
        if lp_genn:
            index_of_net = 1
        else:
            index_of_net = 0
        # index_of_net = 0
        states = [state[0] for state in states]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[index_of_net](states_transformed.to(device))
        valss = torch.tanh(vals)
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = valss.squeeze(-1).tolist()

        return list(zip(pollist, vallist))


    mcts = MCTS_CHESS(env.env.env.env.env.action_spaces['player_0'].n, None, state_evaluator, None, **kwargs)

    def action_value_generator(state, lp_genn):
        mcts.mcts_batch(env, state, num_batches, batch_size, lp_genn)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)


    return run_episode_generic_lp_chess(env, action_value_generator, max_length, max_len_reward,
                                  ptan.actions.ProbabilityActionSelector(),
                                  state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)


def run_chess_ppo(nets: List[torch.nn.Module], env: chessEnv, max_length: int,
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


    def action_value_generator(state, net_to_use):
        states = [state]
        states_transformed = torch.stack(tuple(train_state_rewriter(s[0]) for s in states))
        pols, vals = nets[0](states_transformed.to(device))
        pols1, vals1 = nets[1](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        if net_to_use == 1:
            val = val1
            pollist = pollist1
        elif net_to_use > 1:
            # use the net that gives us the highest value from our current state.
            if val1 > val:
                val = val1
                pollist = pollist1


        return pollist, val


    return run_episode_generic_lp_chess(env, action_value_generator, max_length, max_len_reward,
                                  ptan.actions.ProbabilityActionSelector(),
                                  state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)



def run_checkers_mcts(nets: List[torch.nn.Module], env: checkersEnv, max_length: int,
                         max_len_reward: Union[int, None],
                         device, train_state_rewriter: Callable[[Any], torch.Tensor],
                         state_observer: Callable[[Any], None] = None,lp_gen=False,
                         jensen_shannon=None, output_value=None, num_batches=4, batch_size=5, **kwargs) \
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

    # We will do the action masking here; so the MCTS is only doing legal actions.
    def state_evaluator(states, lp_genn=False):
        if lp_genn:
            index_of_net = 1
        else:
            index_of_net = 0
        # index_of_net = 0
        states = [state['observation'] for state in states]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[index_of_net](states_transformed.to(device))
        valss = torch.tanh(vals)
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = valss.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    # 256 is num actions
    mcts = MCTS_CHECKERS(256, None, state_evaluator, None, **kwargs)

    def action_value_generator(state, lp_genn):
        mcts.mcts_batch(env, state, num_batches, batch_size, lp_genn)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)


    return run_episode_generic_lp_checkers(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)



def run_checkers_ppo(nets: List[torch.nn.Module], env: checkersEnv, max_length: int,
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


    def action_value_generator(state, net_to_use):
        observation = state['observation']
        states = [observation]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))

        pols, vals = nets[0](states_transformed.to(device))
        pols1, vals1 = nets[1](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        if not lp_gen:
            net_to_use = 0

        if net_to_use == 1:
            val = val1
            pollist = pollist1
        elif net_to_use > 1:
            # use the net that gives us the highest value from our current state.
            if val1 > val:
                val = val1
                pollist = pollist1

        return pollist, val

    return run_episode_generic_lp_checkers(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)

episode_runners = {
    "go_no_mcts_noLP": run_go_ppo, # ppo
    "go_no_mcts_LP": run_go_ppo, # ppo LP
    "go_mcts_LP": run_go_mcts_ensemble, # mcts lp
    "go_mcts_noLP": run_go_mcts_ensemble, # mcts no lp

    "chess_noLP_ppo": run_chess_ppo,
    "chess_LP_ppo": run_chess_ppo,
    "chess_mcts_LP": run_chess_mcts_ensemble,
    "chess_mcts_noLP": run_chess_mcts_ensemble,

    "checkers_LP_ppo": run_checkers_ppo,
    "checkers_ppo": run_checkers_ppo,
    "checkers_mcts": run_checkers_mcts,
    "checkers_mcts_LP": run_checkers_mcts,


}

def throwKeyInterr():
    raise KeyboardInterrupt()

def run():
    torch.multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGHUP, throwKeyInterr)


    # TODO: set this to TRUE to load from checkpoint; configure paths right;
    #  BUT do evaluation in a separate main file (main_eval.py) bc it is diff
    LOAD_FROM_CHECKPOINT = False

    for tr_idx in range(0, NUM_RUNS):
        #RUN_POSTFIX = "_tr0_copy"
        RUN_POSTFIX = "_tr" + str(tr_idx) + "_copy"
        #RUN_POSTFIX = ""
        try:
            cman = CheckpointManager(CHECKPOINT_PATH + RUN_POSTFIX, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
        except EOFError:
            cman = CheckpointManager(CHECKPOINT_PATH + RUN_POSTFIX + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)

        # initialize go environment.

        if ENV_TYPE == "go":
            env = goEnvironment(board_size=BOARD_SIZE, board=None, komi=4)
        elif ENV_TYPE == "chess":
            env = chessEnv()
        elif ENV_TYPE == "checkers":
            env = checkersEnv()
        else:
            raise EnvironmentError("specify environment in config that is go, chess, or checkers")

        writer = SummaryWriter(LOG_FOLDER + RUN_POSTFIX)

        train_net_creator, train_rewriter_creator = training_nets[NETWORK]

        net = cman.load("net", train_net_creator(env, **NETWORK_PARAMS),
                        CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

        net_lp = cman.load("net_lp", train_net_creator(env, **NETWORK_PARAMS),
                           CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

        #TODO test LP here when first building LP
        '''
        confusing_input_test = get_confusing_input_pool(net, env, config_file)

        for boardddd in confusing_input_test[0]:
            print(boardddd)
        '''
        net.share_memory()
        net_lp.share_memory()

        nets = [net, net_lp]

        loss_functions = [loss_funcs[LOSS_FUNC](net=nets[0], device=DEVICE, discount=DISCOUNT, **LOSS_PARAMS),
                          loss_funcs[LOSS_FUNC](net=nets[1], device=DEVICE, discount=DISCOUNT, **LOSS_PARAMS)]

        optimizer = cman.load("opt", OPTIMIZER(net.parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                              StateDictLoadHandler())
        optimizer_lp = cman.load("opt_lp", OPTIMIZER(net_lp.parameters(), lr=LEARNING_RATE, **OPTIMIZER_PARAMS),
                                 StateDictLoadHandler())
        optimizers = [optimizer, optimizer_lp]

        # redo for each environment go checkers chess
        train_rewriter = train_rewriter_creator(env)

        train_loop = cman.load("train_loop",
                               RandomReplayTrainingLoop(DISCOUNT, REPLAY_BUFFER, MIN_TRACE_TO_TRAIN, PPO_TRAIN_ROUNDS,
                                                        train_rewriter, writer, DEVICE, game=ENV_TYPE),
                               StateDictLoadHandler())

        if EPISODE_RUNNER_TYPE == "go_no_mcts_LP" or EPISODE_RUNNER_TYPE == "go_mcts_LP" or \
            EPISODE_RUNNER_TYPE == "chess_LP_ppo" or EPISODE_RUNNER_TYPE == "chess_mcts_LP" or \
            EPISODE_RUNNER_TYPE == "checkers_LP_ppo" or EPISODE_RUNNER_TYPE == "checkers_mcts_LP":
            lp_gen = True
        else:
            lp_gen = False

        #TODO hardcode both _nets

        with get_parallel_queue(num_processes=NUM_PROCESSES, episode_runner=episode_runners[EPISODE_RUNNER_TYPE],
                                nets=nets, env=env, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
                                state_observer=None, device=DEVICE,
                                train_state_rewriter=train_rewriter,
                                lp_gen=lp_gen, config_file=config_file, all_work_lp=ALL_WORK_LP,
                                lp_every_x_states=LP_EVERY_X_STATES, lp_states_in_pool=LP_POOL_SIZE, game=ENV_TYPE,
                                **EPISODE_RUNNER_PARAMS) as sim_round_queue:
            while True:
                train_loop(sim_round_queue, loss_functions, optimizers)

                if train_loop.num_rounds % CHECKPOINT_EVERY == 0:
                    save_dict = {
                        "net": nets[0],
                        "net_lp": nets[1],
                        "opt": optimizers[0],
                        "opt_lp": optimizers[1],
                        "train_loop": train_loop,
                    }

                    cman.save(save_dict)

                    if STOP_AFTER and train_loop.global_step > STOP_AFTER:
                        print("STOPPING: step limit " + str(train_loop.global_step) + "/" + str(STOP_AFTER))

                        break




if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
