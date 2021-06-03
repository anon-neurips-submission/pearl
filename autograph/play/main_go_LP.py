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

from autograph.lib.mcts_go import MCTS
from autograph.lib.running_ensemble_go import get_parallel_queue, RandomReplayTrainingLoop, run_episode_generic_lp, run_episode_generic
from autograph.lib.shaping import AutShapingWrapper
from autograph.lib.util import element_add
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler, PickleLoadHandler, TransplantCheckpointManager
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer, NoopCuriosityOptimizer
from autograph.net.maze_constructors import mazenet_v1, mazernd_v1, maze_obs_rewrite_creator
from autograph.net.mine_constructors import minenet_v1, mine_obs_rewriter_creator, minernd_v1, \
    mine_mazenet_v1, mine_mazenet_v1_go, mine_obs_go_rewriter_creator
from autograph.net.misc_constructors import gym_make, no_op_cur_make, basic_net, no_op_make
from lp.LP_general_goEnv import get_confusing_input_pool
math.sqrt(1)  # So that the import isn't optimized away (very useful when setting conditional debug breakpoints)

sys.modules["autograph.lib.mazeenv"] = autograph.lib.envs.mazeenv  # Fix broken pickle loading




optimizers = {
    "Adam": Adam,
    "SGD": SGD
}

env_constructors = {
    "minecraft": MineWorldEnv.from_dict,
    "maze": FuelMazeEnv.from_dict,
    "gym": gym_make
}


def no_op_rewriter(x):
    return torch.Tensor([0.0])


training_nets = {
    "mazenet_v1": (mazenet_v1, maze_obs_rewrite_creator),
    "minenet_v1": (minenet_v1, mine_obs_rewriter_creator),
    "mine_mazenet_v1_go": (mine_mazenet_v1_go, mine_obs_go_rewriter_creator),
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

    aut: dict = config["automaton"]

    LTLF_SPEC = aut["spec"]
    AUT_PARAM_NAMES = [param["name"] for param in aut["params"]]


    def get_func(param: dict):
        func_or_generator = aut_funcs[param["func"]]
        func_params = param.get("params")
        if func_params is None:
            return func_or_generator
        else:
            return func_or_generator(**func_params)


    AUT_PARAM_FUNCS = [get_func(p) for p in aut["params"]]

    AUT_OTHER_PARAMS = {
        "terminate_on_fail": aut.get("terminate_on_fail", True),
        "termination_fail_reward": aut.get("termination_fail_reward", 0),
        "terminate_on_accept": aut.get("terminate_on_accept", False),
        "termination_accept_reward": aut.get("termination_accept_reward", 1)
    }

    AUT_STATS_PARAMS = aut.get("aut_stats_params", dict())

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


    if EPISODE_RUNNER_TYPE == "go_mcts_LP" or EPISODE_RUNNER_TYPE == "go_mcts_noLP":
        LOSS_FUNC = "MCTS"



    # Logging and checkpointing

    LOG_FOLDER = interpolate(args.get("log")) + postfix
    CHECKPOINT_EVERY = int(args["checkpoint_every"])
    CHECKPOINT_PATH = interpolate(args.get("checkpoint")) + postfix

    """
    There are two types of "transplants":
    1. "Old" transplant, this just literally loads the state from the "from" checkpoint instead of creating the state
    from scratch
    2. "Regular" transplant, this is only for the automaton statistics, and it anneals between the imported values and
    the values created during this run."""
    transplant_config = config.get("transplant")
    TRANSPLANT = False
    OLD_TRANSPLANT: Union[bool, List[str]] = False

    if transplant_config:
        TRANSPLANT_FROM = transplant_config["from"]
        if transplant_config.get("fields"):
            OLD_TRANSPLANT = transplant_config["fields"]
        else:
            TRANSPLANT = True
            aut_transplant = transplant_config["automaton"]
            ANNEAL_AUT_TRANSPLANT = aut_transplant["type"]
            ANNEAL_AUT_TRANSPLANT_PARAMS = aut_transplant.get("params", {})

    if CHECKPOINT_PATH:
        LOAD_FROM_CHECKPOINT = args["load_checkpoint"]
        if not os.path.isfile(CHECKPOINT_PATH):
            LOAD_FROM_CHECKPOINT = False
            print("NOTE: no existing checkpoint found, will create new one if checkpoint saving is enabled.")
        else:
            if OLD_TRANSPLANT:
                OLD_TRANSPLANT = False
                print("NOTE: Loading from checkpoint, so transplant disabled")

        SAVE_CHECKPOINTS = args["save_checkpoint"]
    else:
        CHECKPOINT_PATH = None
        LOAD_FROM_CHECKPOINT = False
        SAVE_CHECKPOINTS = False
        if not args["save_checkpoint"]:
            print("WARNING: This run is not being checkpointed! Use --do-not-save-checkpoint to suppress.")

    NUM_PROCESSES = int(args["workers"])
    DEVICE = torch.device(args["device"])


def run_go_mcts_ensemble_LP(nets: List[torch.nn.Module], env: goEnvironment, max_length: int,
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


    mcts = MCTS(env.env.env.env.env.action_space.n, None, state_evaluator, None, **kwargs)

    def action_value_generator(state, lp_genn):
        mcts.mcts_batch(env, state, num_batches, batch_size, lp_genn)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)


    return run_episode_generic_lp(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)


def run_go_mcts_ensemble_noLP(nets: List[torch.nn.Module], env: goEnvironment, max_length: int,
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


    mcts = MCTS(env.env.env.env.env.action_space.n, None, state_evaluator, None, **kwargs)

    def action_value_generator(state, lp_genn):
        mcts.mcts_batch(env, state, num_batches, batch_size, lp_genn)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)


    return run_episode_generic_lp(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=False, jensen_shannon=None, output_valueS=None)




def run_aut_episode_ensemble_deepQ(nets: List[torch.nn.Module], env: AutShapingWrapper, max_length: int,
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
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
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


    return run_episode_generic_lp(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)



def run_go_nomcts_LP(nets: List[torch.nn.Module], env: AutShapingWrapper, max_length: int,
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

        action_mask = state['action_mask']
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
        #net_to_use = 0

        if net_to_use == 1:
            val = val1
            pollist = pollist1
        elif net_to_use > 1:
            # use the net that gives us the highest value from our current state.
            if val1 > val:
                val = val1
                pollist = pollist1

        return pollist, val

    return run_episode_generic_lp(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=lp_gen, jensen_shannon=jensen_shannon, output_valueS=output_value)



def run_go_nomcts_noLP(nets: List[torch.nn.Module], env: AutShapingWrapper, max_length: int,
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

        action_mask = state['action_mask']
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

    return run_episode_generic_lp(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer, lp_gen=False, jensen_shannon=jensen_shannon, output_valueS=output_value)

def run_aut_episode(nets: torch.nn.Module, env: AutShapingWrapper, max_length: int, max_len_reward: Optional[int],
                    device,train_state_rewriter: Callable[[Any], torch.Tensor],
                    state_observer: Callable[[Any], None] = None, render_every_frame=False, lp_gen=False) -> Tuple[
    List[TraceStep], float]:

    def action_value_generator(state, step):
        states = [state]
        states_transformed = torch.stack(tuple(train_state_rewriter(s) for s in states))
        pols, vals = nets[0](states_transformed.to(device))
        pols1, vals1 = nets[1](states_transformed.to(device))
        valss = torch.sigmoid(vals)
        pollist = F.softmax(pols, dim=-1).tolist()[0]

        valss1 = torch.sigmoid(vals1)
        pollist1 = F.softmax(pols1, dim=-1).tolist()[0]

        val = valss.squeeze(-1).tolist()[0]
        val1 = valss1.squeeze(-1).tolist()[0]

        # use the net that gives us the highest value from our current state.
        if val1 > val:
            val = val1
            pollist = pollist1

        if render_every_frame:

            env.render()

        return pollist, val

    # TODO curiosity and automaton bonuses
    return run_episode_generic(env, action_value_generator, max_length, max_len_reward,
                               ptan.actions.ProbabilityActionSelector(),
                               state_observer)

episode_runners = {
    "no_mcts_aut_episode_ensemble": run_aut_episode_ensemble_deepQ,
    "go_no_mcts_noLP": run_go_nomcts_noLP, # ppo
    "go_no_mcts_LP": run_go_nomcts_LP, # ppo LP
    "go_mcts_LP": run_go_mcts_ensemble_LP,
    "go_mcts_noLP": run_go_mcts_ensemble_noLP,
    "mcts_aut_episode_ensemble_eval": run_aut_episode,

}


def run():
    torch.multiprocessing.set_start_method("spawn", force=True)
    signal.signal(signal.SIGHUP, throwKeyInterr)

    LOAD_FROM_CHECKPOINT = False

    for tr_idx in range(0, NUM_RUNS):
        #RUN_POSTFIX = "_tr0_copy"
        RUN_POSTFIX = "_tr" + str(tr_idx) + "_copy"
        #RUN_POSTFIX = ""
        try:
            cman = CheckpointManager(CHECKPOINT_PATH + RUN_POSTFIX, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)
        except EOFError:
            cman = CheckpointManager(CHECKPOINT_PATH + RUN_POSTFIX + "_copy", LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)

        if TRANSPLANT:
            cman = TransplantCheckpointManager(cman, TRANSPLANT_FROM)
            cman.transplant("aut")  # Generating the automaton may not be completely deterministic, we want the same states
        elif OLD_TRANSPLANT:
            cman = TransplantCheckpointManager(cman, TRANSPLANT_FROM)
            for field in OLD_TRANSPLANT:
                cman.transplant(field)

        aut = cman.load("aut", AutomatonSet.from_ltlf(LTLF_SPEC, AUT_PARAM_NAMES), PickleLoadHandler())

        # initialize go environment.
        env = goEnvironment(board_size=BOARD_SIZE, board=None, komi=4)


        writer = SummaryWriter(LOG_FOLDER + RUN_POSTFIX)

        train_net_creator, train_rewriter_creator = training_nets[NETWORK]

        net = cman.load("net", train_net_creator(env, **NETWORK_PARAMS),
                        CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

        net_lp = cman.load("net_lp", train_net_creator(env, **NETWORK_PARAMS),
                           CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)

        #TODO lets test the LP Here

        '''confusing_input_test = get_confusing_input_pool(net, env, config_file)

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
                                                        train_rewriter, writer, DEVICE),
                               StateDictLoadHandler())



        if EPISODE_RUNNER_TYPE == "go_no_mcts_LP" or EPISODE_RUNNER_TYPE == "go_mcts_LP":
            lp_gen = True
        else:
            lp_gen = False

        #TODO hardcode both _nets

        with get_parallel_queue(num_processes=NUM_PROCESSES, episode_runner=episode_runners[EPISODE_RUNNER_TYPE],
                                nets=nets, env=env, max_length=MAX_EPISODE_LEN, max_len_reward=MAX_LEN_REWARD,
                                state_observer=None, device=DEVICE,
                                train_state_rewriter=train_rewriter,
                                lp_gen=lp_gen, config_file=config_file, all_work_lp=ALL_WORK_LP,
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
                        "aut": aut,
                    }

                    cman.save(save_dict)

                    if STOP_AFTER and train_loop.global_step > STOP_AFTER:
                        print("STOPPING: step limit " + str(train_loop.global_step) + "/" + str(STOP_AFTER))

                        break




if __name__ == '__main__':
    multiprocessing.freeze_support()
    run()
