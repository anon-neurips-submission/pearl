import collections
import functools
import multiprocessing.context
import os
import random
from _signal import SIGKILL
from collections import deque
from copy import deepcopy
from typing import Callable, Any, Tuple, List, Union
import json5 as json

import numpy
import psutil
import ptan
import torch
from decorator import contextmanager
from gym import Env
from tensorboardX import SummaryWriter
from torch import multiprocessing as multiprocessing
from torch.nn import functional as F
from torch.optim import Optimizer

from multiprocessing import Queue


from autograph.lib.envs.saveloadenv import SaveLoadEnv
from autograph.lib.loss_functions import LossFunction
from autograph.lib.mcts import MCTS
from autograph.lib.util.trace_info_processor import TraceInfoPreprocessor
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer
from autograph.net.mazenet import Mazenet
from autograph.lib.envs.mineworldenv import MineWorldEnv
from lp.LP_general import get_confusing_input_pool
#from lp.LP_general import get_confusing_input_combined, get_confusing_input_pool
from autograph.lib.shaping import AutShapingWrapper
import cplex


# play steps happen here.
def run_episode_generic_lp(env: AutShapingWrapper,
                           action_value_generator: Callable[[Any, int], Tuple[List[float], float]],
                           max_length: int,
                           max_len_reward: Union[int, None],
                           action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                           state_observer: Callable[[Any], None] = None, render: bool = False, lp_gen=False,
                           jensen_shannon=None,
                           output_valueS=None):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done = False
    if not lp_gen:
        next_state = env.reset()
    else:
        next_state = env.env._get_observation()
    # run the LP and get the confusing, binary constrained tensor according to net and newenv(ironment)
    # we specify we want value minimization to determinte environment; but with many binary constraints
    # value minimzation and policy uniform generation are often the same result

    next_action_raw, next_value = action_value_generator(next_state, 0)
    length = 0

    trace = []

    while not done:
        length += 1

        # Take an action and get results of the action
        state, action_raw, value = next_state, next_action_raw, next_value
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        action_selected = action_selector(numpy.array([action_raw]))[0]
        next_state, reward, done, info = env.step(action_selected)

        if render:
            env.render()

        if state_observer:
            state_observer(next_state)

        done_from_env = 1 if done else 0

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done:
            next_value = 0
        else:
            next_action_raw, next_value = action_value_generator(next_state, length)
            if length >= max_length:
                done = True
                if max_len_reward is not None:
                    next_value = max_len_reward

        trace.append(
            TraceStep(state, value, action_raw, action_selected, next_state, next_value, reward, info, done_from_env))

    return trace, next_value, lp_gen, jensen_shannon, output_valueS


# play steps happen here.
def run_episode_generic(env: AutShapingWrapper, action_value_generator: Callable[[Any, int], Tuple[List[float], float]],
                        max_length: int,
                        max_len_reward: Union[int, None],
                        action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                        state_observer: Callable[[Any], None] = None, render: bool = True):
    """
    Run an episode in an environment
    :param env: The environment to run the episode in
    :param action_value_generator: Given a state and step number, produce some action probabilities and a value estimate
    :param max_length: The step number after which to cut off the episode
    :param action_selector: How to decide the right actions to select given NN "probabilities"
    :param state_observer:A callback function to notify of all states encountered
    :return: A list of steps taken, and the value estimate of the final state
    """
    done = False
    next_state = env.reset()

    # run the LP and get the confusing, binary constrained tensor according to net and newenv(ironment)
    # we specify we want value minimization to determinte environment; but with many binary constraints
    # value minimzation and policy uniform generation are often the same result

    next_action_raw, next_value = action_value_generator(next_state, 0)
    length = 0

    trace = []

    while not done:
        length += 1

        # Take an action and get results of the action
        state, action_raw, value = next_state, next_action_raw, next_value
        # action_probs = F.softmax(action_raw)
        # action_selected = action_selector(action_probs.unsqueeze(0).cpu().numpy())[0]
        # TODO
        summ = sum(action_raw)
        if (summ != 1):
            action_raw1 = list(action / summ for action in action_raw)
            action_raw = action_raw1

        action_selected = action_selector(numpy.array([action_raw]))[0]
        next_state, reward, done, info = env.step(action_selected)

        if render:
            env.render()

        if state_observer:
            state_observer(next_state)

        done_from_env = 1 if done else 0

        # Get the next action/value pair (now instead of beginning of loop so that we easily have access to next value)
        if done:
            next_value = 0
        else:
            next_action_raw, next_value = action_value_generator(next_state, length)
            if length >= max_length:
                done = True
                if max_len_reward is not None:
                    next_value = max_len_reward

        trace.append(
            TraceStep(state, value, action_raw, action_selected, next_state, next_value, reward, info, done_from_env))

    return trace, next_value, False, None, None


def run_episode(net: Mazenet, env: Env, max_length: int, device,
                action_selector: ptan.actions.ActionSelector = ptan.actions.ArgmaxActionSelector(),
                state_observer: Callable[[Any], None] = None) -> Tuple[List[TraceStep], float, bool]:
    """
    Run an episode using the policy from the given network
    :param net: The network whose policy to follow- should have a forward_obs method that returns the action outputs and
    value estimate given a single observation
    :param env: The environment to run the simulation in
    :param max_length: How long the episode should be allowed to run before cutting it off
    :param action_selector: A callable that accepts a list of action probabilities and returns the action to choose
    :param state_observer: A callable that will be called with each state
    :param device: What device to run the network on
    :return: A list of TraceStep tuples and a value estimate of the last state
    """

    def obs_helper(state, step):
        action, value = net.forward_obs(state, device)
        return action.detach(), value.detach().squeeze(-1)

    return run_episode_generic(env, obs_helper, max_length, action_selector, state_observer)


def run_mcts_episode(net: Mazenet, env: SaveLoadEnv, max_length: int, curiosity: ModuleCuriosityOptimizer, device,
                     c_puct,
                     num_batches, batch_size,
                     state_observer: Callable[[Any], None] = None) -> Tuple[List[TraceStep], float, bool]:
    """
    Run an episode using MCTS with curiosity as the action selection
    :param net: The policy/value network
    :param env: The environment to run the simulation in
    :param max_length: When to cut off the simulation
    :param curiosity: Something to calculate the relative "newness" of a state
    :param device: The device to run the simulation on
    :param c_puct: Puct constant of MCTS
    :param num_batches: How many groups of MCTS sims to run
    :param batch_size: How many MCTS sims per group
    :param state_observer: Function to call for every state seen
    :return: A trace and final value estimate
    """

    def curiosity_evaluator(sars):
        states, actions, rewards, next_states = zip(*sars)
        rewards = curiosity.get_curiosity(states, actions, next_states)
        return rewards.tolist()

    def curiosity_trainer(sars):
        states, actions, rewards, next_states = zip(*sars)
        curiosity.train(states, actions, next_states, train_rounds=1)

    def state_evaluator(states):
        states_transformed = torch.stack(tuple(net.rewrite_obs(s) for s in states))
        pols, vals = net(states_transformed.to(device))
        pollist = F.softmax(pols, dim=-1).tolist()
        vallist = vals.squeeze(-1).tolist()

        return list(zip(pollist, vallist))

    mcts = MCTS(env.action_space.n, curiosity_evaluator, state_evaluator, curiosity_trainer, c_puct=c_puct)

    def action_value_generator(state, step):
        mcts.mcts_batch(env, state, num_batches, batch_size)
        probs, values = mcts.get_policy_value(state, 1)
        return probs, max(values)

    return run_episode_generic(env, action_value_generator, max_length, ptan.actions.ProbabilityActionSelector(),
                               state_observer)


def calculate_returns_adv(trace: List[TraceStep], last_value: float, discount: float) \
        -> List[Tuple[float, float, float]]:
    """
    Given a trace, get the discounted returns, advantage, and return advantage value. Advantage is calculated based on the actual
    discounted return versus the value estimate of the state.
    :param trace: The information from a single run
    :param last_value: A value estimate of the final state
    :param discount: The discount factor
    :return: A list of (discounted return, advantage, return_advantage) tuples
    """
    discounted_return = last_value
    next_value = last_value
    returns_adv = []

    for step in reversed(trace):
        discounted_return *= discount
        discounted_return += step.reward
        advantage = step.reward + (discount * next_value) - step.value
        return_advantage = discounted_return - float(step.value)
        returns_adv.append((discounted_return, advantage, return_advantage))

        next_value = step.value

    returns_adv.reverse()

    return returns_adv


def parallel_queue_worker(queue: multiprocessing.SimpleQueue, function_to_run: Callable) -> None:
    """
    Worker that repeatedly adds the result of a function to a queue, but eventually quits when the parent process dies.
    :param queue: The queue to add results to
    :param function_to_run: A function that takes no arguments and produces no results
    :return: When the parent process dies
    """
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    while True:
        result = function_to_run()

        # If the parent process isn't python, that means that it got terminated
        # (and this orphan process got reassigned to init or to some display manager)
        parent = psutil.Process(os.getppid())
        if "python" not in parent.name():
            return

        queue.put(result)
        # pr.dump_stats("performance.cprof")


# this becomes our LP renderenv and return warpper too
class RenderEnvAndReturnWrapper:
    """
    Wraps a given callable such that if the callable accepts an env as a parameter, the env is printed before returning.

    Not a nested function for pickling reasons.
    """

    def __init__(self, func, lp_gen=False, config=None, value_if_true=True, combo=False, device='cuda:0', writer=None,
                 cur_step=None, cur_round=None):
        self.func = func
        self.lp_gen = lp_gen
        self.config_file = config
        self.value_if_true = value_if_true
        self.combo = combo
        self.device = device
        self.writer = writer
        self.cur_round_num = cur_round
        self.cur_step = cur_step

    def __call__(self, env: AutShapingWrapper, net, **kwargs):

        random_gen = True
        jensen = None
        output_value = None
        '''
        # lets try an LP period every so often
        if self.lp_gen:
            if 6000 > self.cur_step > 5000 or \
                    11000 > self.cur_step > 10000 or \
                    16000 > self.cur_step > 15000 or \
                    21000 > self.cur_step > 20000 or \
                    26000 > self.cur_step > 25000 or \
                    31000 > self.cur_step > 30000 or \
                    36000 > self.cur_step > 35000 or \
                    41000 > self.cur_step > 40000 or \
                    46000 > self.cur_step > 45000 or \
                    51000 > self.cur_step > 50000 or \
                    56000 > self.cur_step > 55000:
                self.lp_gen = True
            else:
                self.lp_gen = False

               
        if self.lp_gen:
            # LP every 2/3 games before 30k steps
            # LP every 1/3 games after 30k steps
            if self.cur_step < 10000:
                if self.cur_round_num % 2 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
            elif self.cur_step < 20000:
                if self.cur_round_num % 3 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
            elif self.cur_step < 30000:
                if self.cur_round_num % 4 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
            elif self.cur_step < 40000:
                if self.cur_round_num % 5 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
            elif self.cur_step < 50000:
                if self.cur_round_num % 6 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
            else:
                if self.cur_round_num % 7 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
        '''
        '''
        if self.lp_gen:
            # LP every 2/3 games before 30k steps
            # LP every 1/3 games after 30k steps
            if self.cur_step < 30000:
                if self.cur_round_num % 3 == 0:
                    self.lp_gen = False
                else:
                    self.lp_gen = True
            else:
                if self.cur_round_num % 3 == 0:
                    self.lp_gen = True
                else:
                    self.lp_gen = False
        '''



        print("CUR_GAME(ROUND) = ", self.cur_round_num)
        print("CUR_GLOBAL_STEP = ", self.cur_step)
        print("LP_GEN = ", self.lp_gen)

        if self.lp_gen:

            if self.value_if_true:
                value_det = True
                policy_det = False
            else:
                value_det = False
                policy_det = True
            if self.combo:
                value_det = True
                policy_det = True

            random_gen = False
            try:

                # if not self.combo:
                #    state_tensor = get_confusing_input(net, env.env, self.config_file,
                #                                   policy_det=policy_det, value_det=value_det)
                # else:
                state_tensor, jensen, output_value = get_confusing_input_combined(
                    net, env.env, self.config_file,
                    debug=False, policy=policy_det, value=value_det,
                    device=self.device, writer=self.writer)

                # loads the state into the new environment according to the tensor from LP above

                state = env.env.load_state_from_tensor(state_tensor)
                print('PRE-LP-MIN_VALUE_ENV_CHANGE')

            except cplex.exceptions.errors.CplexSolverError:
                print("No Solution Exists, generating randomly")
                random_gen = True
            env.reset()
            env.env.render()
            if not random_gen:
                new_state = env.env.load_state((state))
                print('POST-LP-MIN_VALUE_ENV_CHANGE')
            else:
                print('No cplex LP solution, doing random generation')
            env.env.render()

        # LP GEN SHOULD ALWAYS BE TRUE W curent architecture.
        # only the render nad reutrn environment does lp gen states
        # @                                                                      added jensen shannon and output values from net for lp passed
        if jensen is None:
            ret = self.func(env=env, net=net, lp_gen=not random_gen, **kwargs)

        else:
            ret = self.func(env=env, net=net, lp_gen=not random_gen, jensen_shannon=jensen, output_value=output_value,
                            **kwargs)
        env.render()
        return ret


# this becomes our LP renderenv and return warpper too
class RenderEnvAndReturnWrapperPool:
    """
    Wraps a given callable such that if the callable accepts an env as a parameter, the env is printed before returning.

    Not a nested function for pickling reasons.
    """

    def __init__(self, func, lp_gen=False, config=None, value_if_true=True, combo=False, device='cuda:0', writer=None,
                 cur_step=None, cur_round=None, lp_state_counts=None, random_late=False):
        self.func = func
        self.lp_gen = lp_gen
        self.config_file = config
        self.value_if_true = value_if_true
        self.combo = combo
        self.device = device
        self.writer = writer
        self.cur_round_num = cur_round
        self.cur_step = cur_step
        # dict keeping track of lp state counts
        self.lp_state_counts = lp_state_counts  # dictionary
        self.randomLate = random_late
        # use queue to hold lp state_counts and grab them

        self.lp_state_queue = collections.deque()

    def __call__(self, env: AutShapingWrapper, net, **kwargs):

        random_gen = True
        jensen = None
        output_value = None

        using_lp = False
        if self.lp_gen:
            # lp every other up to 1000 games and then we do more randoms (every 4 lp)
            if  (self.cur_round_num < 1000 and self.cur_round_num % 3 == 0) or \
                (self.cur_round_num >= 1000 and self.cur_round_num % 4 == 0):
                using_lp = True
            else:
                using_lp = False
            using_lp = True #TODO: set this for the MPI three threaded version


        print("CUR_GAME(ROUND) = ", self.cur_round_num)
        print("CUR_GLOBAL_STEP = ", self.cur_step)
        print("LP_GEN = ", self.lp_gen)
        self.cur_round_num += 1

        if using_lp:

            random_gen = False
            if len(self.lp_state_queue) == 0:
                try:

                    # if not self.combo:
                    #    state_tensor = get_confusing_input(net, env.env, self.config_file,
                    #                                   policy_det=policy_det, value_det=value_det)
                    # else:
                    state_tensors, jensens, output_values = get_confusing_input_pool(
                        net, env.env, self.config_file,
                        debug=False, policy=True, value=True,
                        device=self.device, writer=self.writer)


                    # loads the state into the new environment according to the tensor from LP above

                    for i in range(len(state_tensors)):
                        self.lp_state_queue.append((state_tensors[i], jensens[i], output_values[i]))


                    print('PRE-LP-MIN_VALUE_ENV_CHANGE')

                except cplex.exceptions.errors.CplexSolverError:
                    print("No Solution Exists, generating randomly")
                    random_gen = True


            if len(self.lp_state_queue) != 0 and not random_gen:

                # pull from queue until we get a state thats been used less than 5 times
                while len(self.lp_state_queue) != 0:
                    ret = self.lp_state_queue.popleft()
                    state_tensor, jensen, output_value = ret

                    if state_tensor in self.lp_state_counts and self.lp_state_counts[state_tensor] < 5:
                        self.lp_state_counts[state_tensor] += 1
                        random_gen = False
                        break
                    elif state_tensor not in self.lp_state_counts:
                        self.lp_state_counts[state_tensor] = 1
                        random_gen = False
                        break
                    else:
                        random_gen = True

            else:
                random_gen = True
            env.reset()
            env.env.render()
            if not random_gen:
                state = env.env.load_state_from_tensor(state_tensor)
                new_state = env.env.load_state((state))

                print('POST-LP-MIN_VALUE_ENV_CHANGE')
            else:
                print('No cplex LP solution, doing random generation')
            env.env.render()

        # LP GEN SHOULD ALWAYS BE TRUE W curent architecture.
        # only the render nad reutrn environment does lp gen states
        # @                                                                      added jensen shannon and output values from net for lp passed
        if jensen is None:
            ret = self.func(env=env, net=net, lp_gen=not random_gen, **kwargs)

        else:
            ret = self.func(env=env, net=net, lp_gen=not random_gen, jensen_shannon=jensen, output_value=output_value,
                            **kwargs)
        env.render()
        return ret


# MODIFIED FUNCTION TO OPTIONALLY GENERATE LP CONFUSING INPUT TENSOR IN ONE OF THE WORKERS (WORKER 1).
# NO TIMES OR QUEUES INVOLVED YET
@contextmanager
def get_parallel_queue(num_processes, episode_runner, env, net, device, lp_gen=False, config_file=None,
                       all_work_lp=False,
                       num_rounds=0, global_step=False, **kwargs):
    """
    Create a queue that has a bunch of parallel feeder processes
    :param num_processes: How many feeder processes
    :param episode_runner: A function that produces a trace to be added to the queue
    :param env: The environment to run the simulations in
    :param kwargs: Additional arguments to the episode runner
    :return: The queue that the processes are feeding into, and a list of the created processes
    """
    multiprocessing.set_start_method("spawn", True)

    sim_round_queue = multiprocessing.SimpleQueue()
    processes: List[multiprocessing.context.Process] = []

    cur_round = num_rounds
    cur_step = global_step

    for i in range(num_processes):
        newenv = deepcopy(env)

        render = (i == 0)
        '''
        if render:
            with open(config_file) as f:
                config = json.load(f)

            this_episode_runner = RenderEnvAndReturnWrapper(episode_runner,
                                                            lp_gen=lp_gen, config=config_file,
                                                            value_if_true=True if config['episode_runner']['type']
                                                                                  == 'mcts_aut_episode_min_val' else False)
        else:
            this_episode_runner = episode_runner
        '''
        with open(config_file) as f:
            config = json.load(f)
        if render or all_work_lp:
            this_episode_runner = RenderEnvAndReturnWrapperPool(episode_runner,
                                                            lp_gen=lp_gen, config=config_file,
                                                            value_if_true=True if config['episode_runner']['type']
                                                                                  == 'mcts_aut_episode_min_val' else False,
                                                            combo=True if config['episode_runner']['type']
                                                                          == 'mcts_aut_episode_combined' else False,
                                                            cur_step=cur_step,
                                                            cur_round=cur_round, lp_state_counts=dict(), random_late=global_step)

        else:
            this_episode_runner = episode_runner

        print(render and lp_gen)
        print('WORKER NUM', i)
        mcts_with_args = functools.partial(this_episode_runner, env=newenv, net=net, device=device, **kwargs)

        p = multiprocessing.Process(target=parallel_queue_worker,
                                    args=(sim_round_queue, mcts_with_args))
        p.daemon = True
        p.name = "Worker thread " + str(i)
        p.start()
        processes.append(p)

    try:
        yield sim_round_queue
    finally:
        for p in processes:
            os.kill(p.pid, SIGKILL)


class RandomReplayTrainingLoop:
    """
    A training loop that reads random replays from a replay buffer and trains a loss function based on that
    """

    def __init__(self, discount: float, replay_buffer_len: int, min_trace_to_train: int, train_rounds: int,
                 obs_processor: Callable, writer: SummaryWriter, device, log_LP=False):
        self.device = device
        self.obs_processor = obs_processor
        self.train_rounds = train_rounds
        self.min_trace_to_train = min_trace_to_train
        self.discount = discount
        self.writer = writer
        self.log_LP = log_LP
        self.trace_hooks: List[Callable[[List[TraceReturnStep], float], None]] = []
        self.round_hooks: List[Callable[[int], None]] = []

        self.recent_traces: deque[TraceReturnStep] = deque(maxlen=replay_buffer_len)
        self.global_step = 0
        self.num_rounds = 0
        self.lp_generated_count = 0
        self.lp_set = set()
        self.unique_lp_states = len(self.lp_set)
        self.wins = 0
        self.games = 0
        self.lp_games = 0
        self.lp_wins = 0

    def add_trace_hook(self, hook: Callable[[List[TraceReturnStep], float], None]):
        self.trace_hooks.append(hook)

    def add_round_hook(self, hook: Callable[[int], None]):
        self.round_hooks.append(hook)

    def process_trace(self, trace: List[TraceStep], last_val: float, lp_gen=False):
        """
        Calculate the returns on a trace and add it to the replay buffer
        :param trace: The actions actually taken
        :param last_val: The value estimate of the final state
        """
        ret_adv = calculate_returns_adv(trace, last_val, discount=self.discount)

        trace_adv = [TraceReturnStep(*twi, *ra) for twi, ra in zip(trace, ret_adv)]

        for hook in self.trace_hooks:
            hook(trace_adv, last_val)

        self.recent_traces.extend(trace_adv)
        reward = sum([step.reward for step in trace_adv])

        if reward > 0:
            self.wins += 1
        self.games += 1

        self.writer.add_scalar("run/total_games", self.games, self.global_step)
        self.writer.add_scalar("run/total_wins", self.wins, self.global_step)

        self.writer.add_scalar("run/total_reward", reward, self.global_step)
        self.writer.add_scalar("run/length", len(trace), self.global_step)
        if lp_gen:
            self.writer.add_scalar("lp/lp_total_reward", sum([step.reward for step in trace]), self.global_step)

            self.writer.add_scalar("lp/lp_length", len(trace), self.global_step)
            if reward > 0:
                self.lp_wins += 1
            self.lp_games += 1
        else:
            # WE MUST GO BACK IN POST PROCESSING AND REMOVE STEPS WITH -1....FOR SOME REASON
            # EXTRACTING THE LOGS ERQUIRED a writer input on each step. so we will take out -1 in post processing for graphs
            self.writer.add_scalar("lp/lp_total_reward", -1, self.global_step)
            self.writer.add_scalar("lp/lp_length", -1, self.global_step)

        self.writer.add_scalar("lp/total_games", self.lp_games, self.global_step)
        self.writer.add_scalar("lp/total_wins", self.lp_wins, self.global_step)

        self.global_step += len(trace)

    def train_on_traces(self, traces: List[TraceReturnStep], loss_function: LossFunction, optimizer: Optimizer):
        """
        Minimize a loss function based on a given set of replay steps.
        :param traces:
        :return:
        """
        trinfo = TraceInfoPreprocessor(traces, self.obs_processor, self.device)

        loss, logs = loss_function(trinfo)

        v_loss = logs['loss/value_loss']
        p_loss = logs['loss/policy_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logs["total_loss"] = loss

        return logs

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "recent_traces": self.recent_traces,
            "num_rounds": self.num_rounds
        }

    def load_state_dict(self, state):
        renames = {
            "globalstep": "global_step",
            "numrounds": "num_rounds"
        }

        for key, value in renames.items():
            if key in state and value not in state:
                state[value] = state[key]

        self.global_step = state["global_step"]
        self.num_rounds = state["num_rounds"]
        self.recent_traces = state["recent_traces"]

    def __call__(self, sim_round_queue, loss_function, optimizer):
        self.num_rounds += 1
        # trace is list of traces, last val is 1 if win
        # following three only apply to LP games.
        NUM_ROUND = self.num_rounds
        GLOBAL_STEP = self.global_step
        trace, last_val, lp_generated, jensen_shannon, output_valueS = sim_round_queue.get()

        if lp_generated:
            lp_state = trace[0][0]
            # keep track of unique LP states
            if lp_state not in self.lp_set:
                self.unique_lp_states += 1
                self.lp_set.add(lp_state)
            self.writer.add_scalar("lp/jensenShannon", jensen_shannon, self.global_step)
            self.writer.add_scalar("lp/value_from_net", output_valueS, self.global_step)
        else:
            # GO THROUGH LOGS AND REMOVE ALL THESE...MUST BE HERE FOR SOME REASON LOGGING WONT WORK WITHOUT INCLUDING
            self.writer.add_scalar("lp/jensenShannon", 0.499999, self.global_step)
            self.writer.add_scalar("lp/value_from_net", 0.511111, self.global_step)

        self.process_trace(trace, last_val, lp_generated)

        if len(self.recent_traces) < self.min_trace_to_train:
            return

        for i in range(self.train_rounds):
            rand_traces = random.sample(self.recent_traces, self.min_trace_to_train)
            logs = self.train_on_traces(rand_traces, loss_function, optimizer)

            if i == self.train_rounds - 1:
                for key, value in logs.items():
                    self.writer.add_scalar(key, value, self.global_step)
            self.writer.add_scalar("lp/unique_lp_states", self.unique_lp_states, self.global_step)

        for hook in self.round_hooks:
            hook(self.num_rounds)
