
import cplex
import numpy as np
import matplotlib.pyplot as plt
# keras imports for the dataset and building our neural network

import math
import os
import signal
import sys
from typing import Callable, Any, Tuple, List, Union, Optional

import ptan
import torch
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
from autograph.lib.mcts_aut import MCTSAut, AutStats, ExponentialAnnealedAutStats, UCBAnnealedAutStats
from autograph.lib.running import get_parallel_queue, RandomReplayTrainingLoop, run_episode_generic
from autograph.lib.shaping import AutShapingWrapper
from autograph.lib.util import element_add
from autograph.lib.util.checkpoint_manager import CheckpointManager, StateDictLoadHandler, CombinedLoadHandler, \
    InitZeroLoadHandler, PickleLoadHandler, TransplantCheckpointManager
from autograph.lib.util.trace_return_step import TraceStep, TraceReturnStep
from autograph.net.curiosity.curiosity_optimizer import ModuleCuriosityOptimizer, NoopCuriosityOptimizer
from autograph.net.maze_constructors import mazenet_v1, mazernd_v1, maze_obs_rewrite_creator
from autograph.net.mine_constructors import minenet_v1, mine_obs_rewriter_creator, minernd_v1, mine_mazenet_v1
from autograph.net.misc_constructors import gym_make, no_op_cur_make, basic_net, no_op_make
import json5 as json

DEBUG = True


class MineInfoAutAP:
    def __init__(self, apname: str = None, ap_name: str = None):
        if not (apname or ap_name):
            raise ValueError("Did not provide ap_name to info aut")
        self.name = apname or ap_name

    def __call__(self, action, obs, rew, done, info):
        return self.name in info["atomic_propositions"]


class MineInventoryAP:
    def __init__(self, inventory_item, quantity):
        self.item = inventory_item
        self.quantity = quantity

    def __call__(self, action, obs, rew, done, info):
        return info["inventory"][self.item] == self.quantity


class MineLocationAP:
    def __init__(self, location):
        self.location = tuple(location)

    def __call__(self, action, obs, rew, done, info):
        position, *_ = obs
        return position == self.location


config_file = 'autograph/play/config/mine_woodfactory/simple_eval_envs/large_eval_simple.json'

aut_funcs = {
    "info_aut": MineInfoAutAP,
    "mine_inventory": MineInventoryAP,
    "mine_location": MineLocationAP
}

training_nets = {
    "mazenet_v1": (mazenet_v1, maze_obs_rewrite_creator),
    "minenet_v1": (minenet_v1, mine_obs_rewriter_creator),
    "mine_mazenet_v1": (mine_mazenet_v1, mine_obs_rewriter_creator),
    "basicnet": (basic_net, lambda e: torch.Tensor)
}

env_constructors = {
    "minecraft": MineWorldEnv.from_dict,
    "maze": FuelMazeEnv.from_dict,
    "gym": gym_make
}

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

if "maze" in config:
    maze = config["maze"]

    config["env"] = dict()
    config["env"]["type"] = "maze"
    config["env"]["max_episode_len"] = maze["max_episode_len"]
    del maze["max_episode_len"]
    config["env"]["params"] = maze
    del config["maze"]

env = config["env"]
MAX_EPISODE_LEN = env["max_episode_len"]
MAX_LEN_REWARD = env.get("max_len_reward")
ENV_CONFIG = env["params"]
ENV_TYPE = env["type"]

# Policy training hyperparameters
training: dict = config["training"]

LEARNING_RATE = training["learning_rate"]
REPLAY_BUFFER = training["replay_buffer"]
MIN_TRACE_TO_TRAIN = training["min_trace_to_train"]
PPO_TRAIN_ROUNDS = training["train_rounds"]
NETWORK = training.get("network", "mazenet_v1")
NETWORK_PARAMS = training.get("params", dict())
optimizers = {
    "Adam": Adam,
    "SGD": SGD
}
OPTIMIZER = optimizers[training.get("optimizer")]
OPTIMIZER_PARAMS = training.get("opt_params", {})

# Loss function
loss: dict = config.get("loss")
if loss:
    LOSS_FUNC = loss["type"]
    LOSS_PARAMS = loss.get("params", dict())
else:
    LOSS_FUNC = "MCTS"
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

NETWORK = 'mine_mazenet_v1'

CHECKPOINT_PATH = 'checkpoints/lp_complicate_1_copy'

LOAD_FROM_CHECKPOINT = True

SAVE_CHECKPOINTS = True

DEVICE = 'cuda:0'

cman = CheckpointManager(CHECKPOINT_PATH, LOAD_FROM_CHECKPOINT, SAVE_CHECKPOINTS, device=DEVICE)

orig_env = env_constructors[ENV_TYPE](ENV_CONFIG)

train_net_creator, train_rewriter_creator = training_nets[NETWORK]

net = cman.load("net", train_net_creator(orig_env, **NETWORK_PARAMS),
                   CombinedLoadHandler(StateDictLoadHandler(), InitZeroLoadHandler())).to(DEVICE)
#net = train_net_creator(orig_env, **NETWORK_PARAMS).to(DEVICE)
net.training = False
net.share_memory()


layers = [(name, param.size()) for name, param in net.named_parameters()]



print(layers)
print(len(layers))
#net = load_net("MNIST_digits__avgPool_dense_softmax_together_net.h5")
#results = net.evaluate(X_test, Y_test)
#print("test loss, test acc:", results)
#
# net.summary()
#
#
# ##extractor function:
# extractor = keras.net(inputs=net.inputs,
#                         outputs=[layer.output for layer in net.layers])
#
# # let the samoe of interest be:
# X = X_test[1]
# features = extractor(X.reshape(1,28,28,1))



print('config file and net have been imported')

print('break')


############ number of classes
# number_of_classes = net.layers[-1].output_shape[1]
# print('number of classes = ' , number_of_classes)

number_of_classes = 6

#######################################################################

# build variables indicies based on shape
def build_indicies_dictionary(ls):
    dictionary = {}
    if len(ls) < 2:
        for i in range(ls[0]):
            dictionary[(i)] = (i)
    elif len(ls) < 3:
        for i in range(ls[0]):
            for j in range(ls[1]):
                dictionary[(i, j)] = (i, j)
    else:
        for i in range(ls[0]):
            for j in range(ls[1]):
                for k in range(ls[2]):
                    dictionary[(i, j, k)] = (i, j, k)
    return dictionary


#XX_1_dictionary = build_indicies_dictionary(N_i,N_j,N_k)

#################################################################################### build variable names
#################################################################################### build variable names
X_dictionary_names={}
for layer in layers:
    print(layer)
    l = list(layer[1])
    if len(l) == 1:
        X_dictionary_names[("X_{0}".format(layer[0]))] = tuple(layer[1])
    else:
        X_dictionary_names[("X_{0}".format(layer[0]))] = tuple(layer[1][1:])

#
# add X_input with shape input_shape at the beginning of X_dictionary_names if the net does not have input Layer !!
#input_shape = layers[0][1][1:]
#new_element = {'X_input': input_shape}
#X_dictionary_names = {**new_element,**X_dictionary_names}
#
print('X_dictionary_names = ' , X_dictionary_names)
#
names_string = list(X_dictionary_names.keys())
#names_string = names_string[2:] #this is only for VGG test net 3
print('names_string = ' , names_string)

weight_dims  = list(X_dictionary_names.values())

# input shape depends on maze_shape which gives the first two and the third is in channels
#shape = list(tuple(t[1]) for t in layers)
#shape=[(5,5,5),(32),(32,5,5),(64),(2304),(256),(256),(128),(128),(128),(128),(128),(128),(6),(6)]
print(f'weight_dims = {weight_dims}')


shape = ((6, 6, 5), (5, 5, 32), (5, 5, 32), (4, 4, 64), (4, 4, 64), (1024), (256), (128), (128), (128), (128), (128), (1), (1))

print(f'shapes = {shape}')
################################ BUILD MANUALLY VARIABLE NAMES:

X_0 = {(i,j,k): 'X_0(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[0][0],shape[0][1], shape[0][2]])}
X_1 = {(i,j,k): 'X_1(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[1][0],shape[1][1], shape[1][2]])}
X_2 = {(i,j,k): 'X_2(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[2][0],shape[2][1], shape[2][2]])}
X_3 = {(i,j,k): 'X_3(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[3][0],shape[3][1], shape[3][2]])}
X_4 = {(i,j,k): 'X_4(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[4][0],shape[4][1], shape[4][2]])}
X_5 = {(i): 'X_5(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[5]])}
X_6 = {(i): 'X_6(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[6]])}
X_7 = {(i): 'X_7(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[7]])}
X_8 = {(i): 'X_8(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[8]])}
X_9 = {(i): 'X_9(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[9]])}
X_10 = {(i): 'X_10(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[10]])}

X_11 = {(i): 'X_11(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[11]])}
X_12 = {(i): 'X_12(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[12]])}
#X_13 = {(i): 'X_13(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[13]])}
#X_14 = {(i): 'X_14(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[14]])}
X_13 = {(i): 'X_13(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[13]])}



A = {(i,j): 'A(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}
A_fact = {(i,j): 'A_fact(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}
Agent_pos = {(i,j): 'Agent_pos(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}

Wood_inv = {(i): 'Wood_inv(i{0})'.format(i) for (i) in build_indicies_dictionary([1])}
Tool_inv = {(i): 'Tool_inv(i{0})'.format(i) for (i) in build_indicies_dictionary([1])}


'''
factory_limit = 1
A_factory == 1
factory slice sums to 1

tensor([[[[1., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],   AGENT POSITION
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 1., 0.],  WOOD location
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],   FACTORY location
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0.],   Inv Wood
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],  Inv tool
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.]]]])
          
sum == wood_limit
A_wood...  reflection of # of woods on board 

enforce that the slice of the tensor that is wood adds up to 2
'''

# get params from pytorch
net_params = list(net.parameters())
# [(5, 5, 5), (32,), (32, 5, 5), (64,), (2304,), (256,), (256,),
#  (128,), (128,), (128,), (128,), (6,), (128,), (128,), (128,), (1,)]






#################################################################### start cplex:
problem = cplex.Cplex()

############################################## define whether maximize or minimize
problem.objective.set_sense(problem.objective.sense.minimize)

############################################### add variables with bounds (X_input and output of each layer):
################################################################ this defines BOUNDS and add variables to the cplex problem

# ub from 1 to 2
problem.variables.add(names=list(X_0.values()), lb=[0.0] * len(X_0), ub=[2.0] * len(X_0))

problem.variables.add(names=list(X_1.values()),lb=[-5.0] * len(X_1), ub=[5.0] * len(X_1))
problem.variables.add(names=list(X_2.values()),lb=[   0.0] * len(X_2), ub=[5.0] * len(X_2))

problem.variables.add(names=list(X_3.values()),lb=[   -5.0] * len(X_3), ub=[5.0] * len(X_3))
problem.variables.add(names=list(X_4.values()),lb=[   0.0] * len(X_4), ub=[5.0] * len(X_4))
problem.variables.add(names=list(X_5.values()),lb=[0.0] * len(X_5), ub=[5.0] * len(X_5))

problem.variables.add(names=list(X_6.values()),lb=[-5.0] * len(X_6), ub=[5.0] * len(X_6))

problem.variables.add(names=list(X_7.values()),lb=[0.0] * len(X_7), ub=[5.0] * len(X_7))

problem.variables.add(names=list(X_8.values()),lb=[-5.0] * len(X_8), ub=[5.0] * len(X_8))

problem.variables.add(names=list(X_9.values()),lb=[0.0] * len(X_9), ub=[5.0] * len(X_9))

problem.variables.add(names=list(X_10.values()),lb=[-5.0] * len(X_10), ub=[5.0] * len(X_10))

problem.variables.add(names=list(X_11.values()),lb=[0.0] * len(X_11), ub=[5.0] * len(X_11))

problem.variables.add(names=list(X_12.values()),lb=[-10.0] * len(X_12), ub=[10.0] * len(X_12))

problem.variables.add(names=list(X_13.values()),lb=[0.0] * len(X_13), ub=[10.0] * len(X_13))


######### add the binary varible
problem.variables.add(names=list(A.values()), lb=[0.0] * len(A), ub=[1.0] * len(A))
problem.variables.set_types([(i, problem.variables.type.binary) for i in A.values()])



######### add the binary variable A_factory
problem.variables.add(names=list(A_fact.values()), lb=[0.0] * len(A_fact), ub=[1.0] * len(A_fact))
problem.variables.set_types([(i, problem.variables.type.binary) for i in A_fact.values()])

######### add the binary variable Agent_pos
problem.variables.add(names=list(Agent_pos.values()), lb=[0.0] * len(Agent_pos), ub=[1.0] * len(Agent_pos))
problem.variables.set_types([(i, problem.variables.type.binary) for i in Agent_pos.values()])

######### add the binary variable Wood inv
problem.variables.add(names=list(Wood_inv.values()), lb=[0.0] * len(Wood_inv), ub=[2.0] * len(Wood_inv))
problem.variables.set_types([(i, problem.variables.type.binary) for i in Wood_inv.values()])

######### add the binary variable Tool inv
problem.variables.add(names=list(Tool_inv.values()), lb=[0.0] * len(Tool_inv), ub=[3.0] * len(Tool_inv))
problem.variables.set_types([(i, problem.variables.type.binary) for i in Tool_inv.values()])

#problem.variables.add(names=list(X_13.values()),lb=[0.0] * len(X_13), ub=[5.0] * len(X_13))

#problem.variables.add(names=list(X_14.values()),lb=[-5.0] * len(X_14), ub=[5.0] * len(X_14))

#problem.variables.add(names=list(X_15.values()),lb=[0.0] * len(X_15), ub=[1.0] * len(X_15))



####################################################################### OBJECTIVES

### all relus
#
#
problem.objective.set_linear(list(zip(list(X_2.values()), [1.0]  * len(X_2 ))))
problem.objective.set_linear(list(zip(list(X_4.values()), [1.0]  * len(X_4 ))))
problem.objective.set_linear(list(zip(list(X_7.values()), [1.0]  * len(X_7 ))))
problem.objective.set_linear(list(zip(list(X_9.values()), [1.0]  * len(X_9 ))))
problem.objective.set_linear(list(zip(list(X_11.values()), [1.0] * len(X_11))))
problem.objective.set_linear(list(zip(list(X_13.values()), [1.0] * len(X_13))))


#####################################################  CONSTRAINTS:


#############################################################################################################################
#############################################     conv                    ###################################################
#############################################################################################################################
'''
X_out = X_1  # this is the output of the (conv) layer
X_in = X_0  # this is the input to the conv layer
lay = 0
# size(input) /= size(output) in the case of a conv layer
shape_out = shape[1]
shape_in  = shape[0]
# get weights and biases
#
# below needs to be pulled out from the pytorch model (torch here)
# W_conv_arr = model.layers[lay].get_weights()[0]
W_conv_arr = net_params[0]
# W_conv_arr = np.ones(shape=(1,1,5,32))
b_conv_arr = net_params[1]
# get conv filter parameters:
shape_W = W_conv_arr.shape

# get conv filters parameters:
strides = 1
pool_size_W = W_conv_arr.shape[2]
pool_size_H = W_conv_arr.shape[3]
pool_size_D = W_conv_arr.shape[1]
if True:
    print(f"{pool_size_W},{pool_size_H},{pool_size_D}")
number_of_filters = W_conv_arr.shape[0]

# for every filter in the conv layer
for nnn in range((number_of_filters)):
    # get the nth filter
    # we want this to be shape 2, 2, 5
    W_nth = W_conv_arr[nnn, :, :, :]
    #W_nth = W_conv_arr[:, :, :,nnn]
    #print(W_nth)
    #print('n = ', nnn, 'W_nth shape = ', W_nth.shape)
    W_nth = W_nth.reshape(pool_size_W, pool_size_H, pool_size_D)
    #print('n = ', nnn, 'W_nth shape = ', W_nth.shape)

    #print("X_in is ", X_in)
    #print("X_out is ", X_out)
    # for every i,j \in I_out X J_out
    for i in range((shape_out[0])):
        for j in range((shape_out[1])):
            # get the portion of input that will be multiplied
            input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
            input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]

            # do the output
            lin_expr_vars_lhs = [X_out[(i, j, nnn)]]
            lin_expr_vals_lhs = [1.0]

            #print("INPUT I ", input_i)
            #print("INPUT J ", input_j)
            #print("^^^^^^^^^^^^^^^^^^^^^")

            #b_conv_arr[nnn]
            # logger
            #print('output indicies: ',i,j,nnn,'filter number = ',nnn,'sum of weights = ',np.sum(W_nth),' bias: ',b_conv_arr[nnn],' input indicies: ', range(input_i[0][0], input_i[0][1] + 1), range(input_j[0][0], input_j[0][1] + 1))
            # loop to do the summation
            for iii in range(input_i[0][0],input_i[0][1]+1):
                for jjj in range(input_j[0][0],input_j[0][1]+1):
                    for kkk in range(pool_size_D):
                        #print((iii, jjj, kkk))

                        if True is False:
                            if (iii, jjj, kkk) in X_in: lin_expr_vars_lhs.append(X_in[(iii, jjj, kkk)])
                            else: continue

                        lin_expr_vars_lhs.append(X_in[(iii, jjj, kkk)])


                        a = round(W_nth[iii - input_i[0][0], jjj - input_j[0][0], kkk].item(),4)
                        lin_expr_vals_lhs.append(-a)
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[round(b_conv_arr[nnn].item(),4)],
                names=["(conv_1)_"])


#############################################################################################################################

"""CONSTRAINTS (conv_2) """
# this is for X_3 = conv(X_2)

# we need X_in, X_out, shape_in, shape_out, weights, and biases

X_out = X_3  # this is the output of the (conv) layer
X_in = X_2  # this is the input to the conv layer
lay = 0
# size(input) /= size(output) in the case of a conv layer
shape_out = shape[3]
shape_in  = shape[2]
# get weights and biases
#
# below needs to be pulled out from the pytorch model (torch here)
# below needs to be pulled out from the pytorch model (torch here)
#W_conv_arr = model.layers[lay].get_weights()[0]
W_conv_arr = net_params[2]
b_conv_arr = net_params[3]
# W_conv_arr = np.ones(shape=(1,1,32,64))
# b_conv_arr = np.ones(shape=(64,1))
# get conv filter parameters:
shape_W = W_conv_arr.shape
#print(shape_W)
#print(f'X_in is {X_in}')

# get conv filters parameters:
#strides = model.layers[lay].strides[0]
strides = 1



# CHECK THESE POOL SIZES
# WHAT IS THE RELATIONSHIP BETWEEN POOL SIZE AND KERNEL/PADDING
pool_size_W = W_conv_arr.shape[2]
pool_size_H = W_conv_arr.shape[3]
pool_size_D = W_conv_arr.shape[1]
if True:
    print(f"{pool_size_W},{pool_size_H},{pool_size_D}")

number_of_filters = W_conv_arr.shape[0]

# for every filter in the conv layer
for nnn in range((number_of_filters)):
    # get the nth filter
    W_nth = W_conv_arr[nnn, :, :, :]
    print(W_nth.shape)
    # print('n = ', nnn, 'W_nth shape = ', W_nth.shape)
    W_nth = W_nth.reshape(pool_size_W, pool_size_H, pool_size_D)

    # for every i,j \in I_out X J_out
    for i in range((shape_out[0])):
        for j in range((shape_out[1])):
            # get the portion of input that will be multiplied
            input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
            input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]
            #print("INPUT I ", input_i)
            #print("INPUT J ", input_j)
            #print("^^^^^^^^^^^^^^^^^^^^^")

            # do the output
            lin_expr_vars_lhs = [X_out[(i, j, nnn)]]
            lin_expr_vals_lhs = [1.0]

            #b_conv_arr[nnn]

            # logger
            #print('output indicies: ',i,j,nnn,'filter number = ',nnn,'sum of weights = ',np.sum(W_nth),' bias: ',b_conv_arr[nnn],' input indicies: ', range(input_i[0][0], input_i[0][1] + 1), range(input_j[0][0], input_j[0][1] + 1))
            # loop to do the summation
            for iii in range(input_i[0][0],input_i[0][1]+1):
                for jjj in range(input_j[0][0],input_j[0][1]+1):
                    for kkk in range(pool_size_D):
                        lin_expr_vars_lhs.append(X_in[(iii, jjj, kkk)])
                        a = round(W_nth[iii - input_i[0][0], jjj - input_j[0][0], kkk].item(),4)
                        lin_expr_vals_lhs.append(-a)
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[round(b_conv_arr[nnn].item(),4)],
                names=["(conv_1)_"])



#############################################################################################################################
#############################################RELU                         ###################################################
#############################################################################################################################
"""CONSTRAINTS (ReLU_1_1) """
#  X_2 >= X_1
X_out = X_2 # this is the output of the Relu
X_in  = X_1 # this is the input to the Relu
shape_ = shape[1]
for i in range(shape_[0]):
    for j in range(shape_[1]):
        for k in range(shape_[2]):
            lin_expr_vars_lhs = [X_out[(i, j, k)]]
            lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

            lin_expr_vars_rhs = [X_in[(i, j, k)]]
            lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                                senses=['G'],
                                rhs=[0.0],
                                names=["(ReLU_1_1)_"])
"""CONSTRAINTS (ReLU_1_2) """
#  X_2 >= 0
X_out = X_2 # this is the output of the Relu
X_in  = X_1 # this is the input to the Relu
shape_ = shape[1]
for i in range(shape_[0]):
    for j in range(shape_[1]):
        for k in range(shape_[2]):
            lin_expr_vars_lhs = [X_out[(i, j, k)]]
            lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

            # lin_expr_vars_rhs = [X_in[(i, j, k)]]
            # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                                senses=['G'],
                                rhs=[0.0],
                                names=["(ReLU_1_2)_"])



#############################################################################################################################

"""CONSTRAINTS (ReLU_2_1) """
#  X_2 >= X_1
X_out = X_4 # this is the output of the Relu
X_in  = X_3 # this is the input to the Relu
shape_ = shape[4]
for i in range(shape_[0]):
    for j in range(shape_[1]):
        for k in range(shape_[2]):
            lin_expr_vars_lhs = [X_out[(i, j, k)]]
            lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

            lin_expr_vars_rhs = [X_in[(i, j, k)]]
            lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                                senses=['G'],
                                rhs=[0.0],
                                names=["(ReLU_2_1)_"])
"""CONSTRAINTS (ReLU_2_2) """
#  X_2 >= 0
X_out = X_4 # this is the output of the Relu
#X_in  = X_1 # this is the input to the Relu
shape_ = shape[4]
for i in range(shape_[0]):
    for j in range(shape_[1]):
        for k in range(shape_[2]):
            lin_expr_vars_lhs = [X_out[(i, j, k)]]
            lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

            # lin_expr_vars_rhs = [X_in[(i, j, k)]]
            # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

            problem.linear_constraints.add(
                                lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                                senses=['G'],
                                rhs=[0.0],
                                names=["(ReLU_2_2)_"])



#############################################################################################################################


"""CONSTRAINTS (ReLU_3_1) """
#  X_2 >= X_1
X_out = X_7 # this is the output of the Relu
X_in  = X_6 # this is the input to the Relu
shape_ = shape[7]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    lin_expr_vars_rhs = [X_in[(i)]]
    lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                        senses=['G'],
                        rhs=[0.0],
                        names=["(ReLU_3_1)_"])
"""CONSTRAINTS (ReLU_3_2) """
#  X_2 >= 0
X_out = X_7 # this is the output of the Relu
#X_in  = X_1 # this is the input to the Relu
shape_ = shape[7]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    # lin_expr_vars_rhs = [X_in[(i, j, k)]]
    # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                        senses=['G'],
                        rhs=[0.0],
                        names=["(ReLU_3_2)_"])



#############################################################################################################################


"""CONSTRAINTS (ReLU_4_1) """
#  X_2 >= X_1
X_out = X_9 # this is the output of the Relu
X_in  = X_8 # this is the input to the Relu
shape_ = shape[9]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    lin_expr_vars_rhs = [X_in[(i)]]
    lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                        senses=['G'],
                        rhs=[0.0],
                        names=["(ReLU_4_1)_"])
"""CONSTRAINTS (ReLU_4_2) """
#  X_2 >= 0
X_out = X_9 # this is the output of the Relu
#X_in  = X_1 # this is the input to the Relu
shape_ = shape[9]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    # lin_expr_vars_rhs = [X_in[(i, j, k)]]
    # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                        senses=['G'],
                        rhs=[0.0],
                        names=["(ReLU_4_2)_"])

#############################################################################################################################

# CONSTRAINTS (ReLU_5_1) 
#  X_2 >= X_1
X_out = X_11 # this is the output of the Relu
X_in  = X_10 # this is the input to the Relu
shape_ = shape[11]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    lin_expr_vars_rhs = [X_in[(i)]]
    lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                        senses=['G'],
                        rhs=[0.0],
                        names=["(ReLU_5_1)_"])

# CONSTRAINTS (ReLU_5_2)
#  X_2 >= 0
X_out = X_11 # this is the output of the Relu
#X_in  = X_1 # this is the input to the Relu
shape_ = shape[11]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    # lin_expr_vars_rhs = [X_in[(i, j, k)]]
    # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                        senses=['G'],
                        rhs=[0.0],
                        names=["(ReLU_5_2)_"])

#############################################################################################################################

'''

#CONSTRAINTS (ReLU_6_1) 
#  X_2 >= X_1
X_out = X_13  # this is the output of the Relu
X_in = X_12  # this is the input to the Relu
shape_ = shape[13]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    lin_expr_vars_rhs = [X_in[(i)]]
    lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
        senses=['G'],
        rhs=[0.0],
        names=["(ReLU_6_1)_"])

#CONSTRAINTS (ReLU_6_2) 
#  X_2 >= 0
X_out = X_13  # this is the output of the Relu
# X_in  = X_1 # this is the input to the Relu
shape_ = shape[13]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)



    problem.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
        senses=['G'],
        rhs=[0.0],
        names=["(ReLU_6_2)_"])




#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


#############################################################################################################################
#################################################### dense here   ###########################################################
#############################################################################################################################
'''
"""CONSTRAINTS (den_1)"""

# we need X_in, X_out, shape_in, shape_out, weights, and biases


W_dense_arr = net_params[4]
# (torch here)
# W_dense_arr = np.ones(shape=(6400,256))
#W_dense_arr.reshape((256, 1024))
b_dense_arr = net_params[5]
# (torch here)
# b_dense_arr = np.ones(shape=(256))
# make the biases an array
X_out = X_6     # this is the output of the FC layer
X_in  = X_5

shape_   = shape[6  ] # shape of the output of the FC  layer
shape_in = shape[5  ] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[i, :]

    # this loop is for the dot product (shape of input)
    for j in range(shape_in):

        lin_expr_vars_lhs.append(X_in[(j)])
        a = round(-WW[j].item(),4)
        lin_expr_vals_lhs.append(a)

    bb = b_dense_arr[i]

    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(bb.item(),4)],
                    names=["(den_1)_"])


#############################################################################################################################

"""CONSTRAINTS (den_2)"""

# we need X_in, X_out, shape_in, shape_out, weights, and biases


W_dense_arr = net_params[6]
b_dense_arr = net_params[7] # make the biases an array

#W_dense_arr = net.layers[4].get_weights()[0]
# (torch here)
#W_dense_arr = np.ones(shape=(256,128))

#b_dense_arr = net.layers[4].get_weights()[1]
# (torch here)
#b_dense_arr = np.ones(shape=(128))

X_out = X_8     # this is the output of the FC layer
X_in  = X_7

shape_   = shape[8  ] # shape of the output of the FC  layer
shape_in = shape[7  ] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[i,:]

    # this loop is for the dot product (shape of input)
    for j in range(shape_in):

        lin_expr_vars_lhs.append(X_in[(j)])
        a = round(-WW[j].item(),4)
        lin_expr_vals_lhs.append(a)

    bb = b_dense_arr[i]

    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(bb.item(),4)],
                    names=["(den_2)_"])


#############################################################################################################################

"""CONSTRAINTS (den_3)"""

# we need X_in, X_out, shape_in, shape_out, weights, and biases

# (torch here)
#W_dense_arr = net.layers[4].get_weights()[0]
#b_dense_arr = net.layers[4].get_weights()[1]         # make the biases an array


W_dense_arr = net_params[12]
b_dense_arr = net_params[13]
# (torch here)
#W_dense_arr = np.ones(shape=(128,128))

#b_dense_arr = net.layers[4].get_weights()[1]
# (torch here)
#b_dense_arr = np.ones(shape=(128))


X_out = X_10     # this is the output of the FC layer
X_in  = X_9

shape_   = shape[10] # shape of the output of the FC  layer
shape_in = shape[9] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[i,:]

    # this loop is for the dot product (shape of input)
    for j in range(shape_in):

        lin_expr_vars_lhs.append(X_in[(j)])
        a = round(-WW[j].item(),4)
        lin_expr_vals_lhs.append(a)

    bb = b_dense_arr[i]

    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(bb.item(),4)],
                    names=["(den_3)_"])

#############################################################################################################################

#############################################################################################################################



#############################################################################################################################
##################################################### flatten   ##################################################
#############################################################################################################################
'''


#CONSTRAINTS (den_4)

# we need X_in, X_out, shape_in, shape_out, weights, and biases

# (torch here)
#W_dense_arr = net.layers[4].get_weights()[0]
#b_dense_arr = net.layers[4].get_weights()[1]         # make the biases an array


W_dense_arr = net_params[14]
# (torch here)
#W_dense_arr = np.ones(shape=(128,128))

b_dense_arr = net_params[15]
# (torch here)
#b_dense_arr = np.ones(shape=(128))

X_out = X_12     # this is the output of the FC layer
X_in  = X_11

shape_   = shape[12] # shape of the output of the FC  layer
shape_in = shape[11] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[i]

    # this loop is for the dot product (shape of input)
    for j in range(shape_in):

        lin_expr_vars_lhs.append(X_in[(j)])
        a = round(-WW[j].item(),4)
        lin_expr_vals_lhs.append(a)

    bb = b_dense_arr[i]

    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(bb.item(),4)],
                    names=["(den_4)_"])


'''
"""CONSTRAINTS (Fltt)"""
# X_5 = flatten(X_4)

X_out = X_5 # this is the output of the Flatten
X_in  = X_4 # this is the input to the Flatten
shape_   = shape[5] # shape of the output of the flatten  layer
shape_in = shape[4] # shape of the input  of the flatten  layer
# ini
l = 0
for i in range(shape_in[0]):
    for j in range(shape_in[1]):
        for k in range(shape_in[2]):
            lin_expr_vars_lhs = [X_in[(i,j,k)]]
            lin_expr_vals_lhs = [1.0]
            lin_expr_vars_rhs = [X_out[(l)]]
            lin_expr_vals_rhs = [-1.0]
            l = l+1
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs,
                                           val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                senses=['E'],
                rhs=[0.0],
                names=["(Fltt)_"])
'''
#############################################################################################################################



#############################################################################################################################
##################################################### constraint (v)(vi)   ##################################################
#############################################################################################################################



#############################################################################################################################

#### hard code the input for debugging purposes
# """CONSTRAINTS (fix_input) """
#
# for i in range(shape[0][0]):
#     for j in range(shape[0][1]):
#         for k in range(shape[0][2]):
#             lin_expr_vars_lhs = [X_0[(i,j,k)]]
#             lin_expr_vals_lhs = [1.0]
#             problem.linear_constraints.add(
#                 lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
#                 senses=['E'],
#                 rhs=[round(X[i,j,k].item(),4)],
#                 names=["(fix_input)_"])


#############################################################################################################################
##################################################### constraint (v)(vi)   ##################################################
#############################################################################################################################

print('break')


# radius from optimal solution. smaller means longer to find solution
#CONSTRAINTS (Constraint_X_LFC)
lin_expr_vars_lhs=[]
lin_expr_vals_lhs=[]

lin_expr_vars_lhs.append(X_13[(0)])
lin_expr_vals_lhs.append(1.0)

# we want that RHS to be less than 0.5 (value)
problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs,  val=lin_expr_vals_lhs)],
                        senses=['L'],
                        rhs=[0.2],
                        names=["(X_LFC_fromObj)_"])

#############################################################################################################################


#############################################################################################################################
##################################################### constraint BINARY !!!!   ##################################################
#############################################################################################################################

# WE HAVE ALREADY DEFINED A BINARY CONSTRAINT A which the same size as the input tensor
'''
# \sum(A) = 100 ;
lin_expr_vars_lhs=[]
lin_expr_vals_lhs=[]

"""CONSTRAINTS (Constraint_binary_1"""
#\sum(A) = 50 ; only 50 entries of the binary tensor is one,
X_out = A

shape_out   = shape[0]

for i in range(shape_out[0]):
    for j in range(shape_out[1]):
        lin_expr_vars_lhs.append(X_out[(i, j)])
        lin_expr_vals_lhs.append(1)

        # sum of list equal to 2
problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[6.0],
                names=["(Constraint_binary_1"])



"""CONSTRAINTS (Constraint_binary_2"""
# picking some input indicies in which X = A ;
# for example X_in(0,0,0) and X_in(0,0,1) and X_in(0,0,2) shoould equal A

# special_input_indicies = [(0,0,0),(0,0,1),(0,0,2)]

X_out = A
X_in  = X_0 # this is the input to the Flatten
# shape_out   = shape[0] # shape of the output of the flatten  layer
# shape_iin   = shape[0] # shape of the input  of the flatten  layer


for i in range(0, 6):
    for j in range(0, 6):

        # i = special_input_indicies[l][0]
        # j = special_input_indicies[l][1]
        # k = special_input_indicies[l][2]

        lin_expr_vars_1 = [X_out[(i, j)]]
        lin_expr_vals_1 = [-1.0] * len(lin_expr_vars_1)
        # hard coded 1 for second slice
        lin_expr_vars_2 = [X_in[(i, j, 1)]]
        lin_expr_vals_2 = [1.0] * len(lin_expr_vars_2)

        aa = lin_expr_vars_1 + lin_expr_vars_2
        bb = lin_expr_vals_1 + lin_expr_vals_2

        problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(aa, bb)],
                        senses=['E'],
                        rhs=[0.0],
                        names=["(Constraint_binary_2"])

"""CONSTRAINTS (Constraint_binary_2_1"""
#\sum(A) = 50 ; only 50 entries of the binary tensor is one,

lin_expr_vars_lhs=[]
lin_expr_vals_lhs=[]

X_out = A_fact

shape_out   = shape[0]

for i in range(shape_out[0]):
    for j in range(shape_out[1]):
        lin_expr_vars_lhs.append(X_out[(i, j)])
        lin_expr_vals_lhs.append(1)

        # sum of list equal to 2


problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[2.0],
                names=["(Constraint_binary_2_1"])



"""CONSTRAINTS (Constraint_binary_2_2"""
# picking some input indicies in which X = A ;
# for example X_in(0,0,0) and X_in(0,0,1) and X_in(0,0,2) shoould equal A

# special_input_indicies = [(0,0,0),(0,0,1),(0,0,2)]

X_out = A_fact
X_in  = X_0 # this is the input to the Flatten
# shape_out   = shape[0] # shape of the output of the flatten  layer
# shape_iin   = shape[0] # shape of the input  of the flatten  layer


for i in range(0, 6):
    for j in range(0, 6):

        # i = special_input_indicies[l][0]
        # j = special_input_indicies[l][1]
        # k = special_input_indicies[l][2]

        lin_expr_vars_1 = [X_out[(i, j)]]
        lin_expr_vals_1 = [-1.0] * len(lin_expr_vars_1)

        # hard coded 2 for third FACTORY location slice
        lin_expr_vars_2 = [X_in[(i, j, 2)]]
        lin_expr_vals_2 = [1.0] * len(lin_expr_vars_2)

        aa = lin_expr_vars_1 + lin_expr_vars_2
        bb = lin_expr_vals_1 + lin_expr_vals_2

        problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(aa, bb)],
                        senses=['E'],
                        rhs=[0.0],
                        names=["(Constraint_binary_2_2"])



"""CONSTRAINTS (Constraint_binary_3_1"""
#\sum(A) = 50 ; only 50 entries of the binary tensor is one,

lin_expr_vars_lhs=[]
lin_expr_vals_lhs=[]

X_out = Agent_pos

shape_out   = shape[0]

for i in range(shape_out[0]):
    for j in range(shape_out[1]):
        lin_expr_vars_lhs.append(X_out[(i, j)])
        lin_expr_vals_lhs.append(1)

        # sum of list equal to 2


problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[1.0],
                names=["(Constraint_binary_3_1"])



"""CONSTRAINTS (Constraint_binary_3_2"""
# picking some input indicies in which X = A ;
# for example X_in(0,0,0) and X_in(0,0,1) and X_in(0,0,2) shoould equal A

# special_input_indicies = [(0,0,0),(0,0,1),(0,0,2)]

X_out = Agent_pos
X_in  = X_0 # this is the input to the Flatten
# shape_out   = shape[0] # shape of the output of the flatten  layer
# shape_iin   = shape[0] # shape of the input  of the flatten  layer


for i in range(0, 6):
    for j in range(0, 6):

        # i = special_input_indicies[l][0]
        # j = special_input_indicies[l][1]
        # k = special_input_indicies[l][2]

        lin_expr_vars_1 = [X_out[(i, j)]]
        lin_expr_vals_1 = [-1.0] * len(lin_expr_vars_1)

        # hard coded 0 for agent position  slice
        lin_expr_vars_2 = [X_in[(i, j, 0)]]
        lin_expr_vals_2 = [1.0] * len(lin_expr_vars_2)

        aa = lin_expr_vars_1 + lin_expr_vars_2
        bb = lin_expr_vals_1 + lin_expr_vals_2

        problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(aa, bb)],
                        senses=['E'],
                        rhs=[0.0],
                        names=["(Constraint_binary_3_2"])


"""CONSTRAINTS (Constraint_binary_4_1"""
#\sum(A) = 50 ; only 50 entries of the binary tensor is one,



X_out = X_0
X_in = Wood_inv

shape_out   = shape[0]
lin_expr_vars_1 = []
lin_expr_vals_1 = []

lin_expr_vars_2 = []
lin_expr_vals_2 = []

for i in range(shape_out[0]):

    for j in range(shape_out[1]):
        lin_expr_vars_1 = [X_out[(i, j, 3)]]
        lin_expr_vals_1 = [1.0] * len(lin_expr_vars_1)


        # wood inv slice
        lin_expr_vars_2 = [X_in[(0)]]
        lin_expr_vals_2 = [-1.0] * len(lin_expr_vars_2)
        # sum of list equal to 2


        problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(lin_expr_vars_1 + lin_expr_vars_2, lin_expr_vals_1 + lin_expr_vals_2)],
                        senses=['E'],
                        rhs=[0],
                        names=["(Constraint_binary_4_1"])

"""CONSTRAINTS (Constraint_binary_5_1"""
#\sum(A) = 50 ; only 50 entries of the binary tensor is one,



X_out = X_0
X_in = Tool_inv

shape_out   = shape[0]
lin_expr_vars_1 = []
lin_expr_vals_1 = []

lin_expr_vars_2 = []
lin_expr_vals_2 = []

for i in range(shape_out[0]):

    for j in range(shape_out[1]):
        lin_expr_vars_1 = [X_out[(i, j, 4)]]
        lin_expr_vals_1 = [1.0] * len(lin_expr_vars_1)


        # wood inv slice
        lin_expr_vars_2 = [X_in[(0)]]
        lin_expr_vals_2 = [-1.0] * len(lin_expr_vars_2)
        # sum of list equal to 2


        problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(lin_expr_vars_1 + lin_expr_vars_2, lin_expr_vals_1 + lin_expr_vals_2)],
                        senses=['E'],
                        rhs=[0],
                        names=["(Constraint_binary_5_1"])

#############################################################################################################################
'''


#### try to print either i,j,k mode or only certaint contraints, bounds, or only objectives
#problem.write( filename='MNIST_digits_.lp')

### this is only used for MIP (Mixed Integer Programming)
problem.parameters.mip.tolerances.integrality.set(1e-4)
#problem.parameters.mip.tolerances.mipgap.set(0.01)
#problem.parameters.mip.tolerances.absmipgap.set(0.01)
problem.parameters.mip.tolerances.mipgap.set(1e-4)
problem.parameters.mip.tolerances.absmipgap.set(1e-4)

problem.write( filename='constraint_check.lp')
problem.solve()



solutionstatus = problem.solution.status[problem.solution.get_status()]
print('LP STATUS: ' , solutionstatus)
print("Solution value  = ", problem.solution.get_objective_value())


# initialize numpy array of zeros to which we map our confusing output dictionary
confusing_output = np.zeros(shape=(5, 6, 6))

# pulling up the generated input image from the LP
temp = {k: problem.solution.get_values(id) for (k, id) in X_0.items()}
print("PRINTING X_0")

# manual reshaping
for (key, value) in temp.items():
    confusing_output[key[2], key[0], key[1]] = value

    if key[2] == 1:
        pass
        # print(f'key: {key} ... value: {value}')






X_0_1d = np.array(list(temp.values()))
print('Solution of ', 'X_0', ' = ', temp)
# pulling up the generated input image from the LP
temp = {k: problem.solution.get_values(id) for (k, id) in A.items()}
A_0 = np.array(list(temp.values()))
print('Solution of ', 'A', ' = ', temp)

# temp = {k: problem.solution.get_values(id) for (k, id) in X_1.items()}
# X_1_1d = np.array(list(temp.values()))
# print('Solution of ', 'X_1', ' = ', temp)
#
# temp = {k: problem.solution.get_values(id) for (k, id) in X_2.items()}
# X_2_1d = np.array(list(temp.values()))
# print('Solution of ', 'X_2', ' = ', temp)
#
# temp = {k: problem.solution.get_values(id) for (k, id) in X_3.items()}
# X_3_1d = np.array(list(temp.values()))
# print('Solution of ', 'X_3', ' = ', temp)
#
# temp = {k: problem.solution.get_values(id) for (k, id) in X_4.items()}
# X_4_1d = np.array(list(temp.values()))
# print('Solution of ', 'X_4', ' = ', temp)
#
# temp = {k: problem.solution.get_values(id) for (k, id) in X_5.items()}
# X_5_1d = np.array(list(temp.values()))
# print('Solution of ', 'X_5', ' = ', temp)


# this is the output the of the last layer from the LP
temp = {k: problem.solution.get_values(id) for (k, id) in X_12.items()}
X_15_1d = np.array(list(temp.values()))
print('Solution of ', 'X_12', ' = ', temp)

plt.figure()
plt.subplot(7,1,1)
plt.title('input 1D')
plt.plot(X_0_1d)
#plt.plot(X.reshape(28*28),'r')
# plt.subplot(7,1,2)
# plt.title('output of conv')
# plt.plot(X_1_1d)
# #plt.plot(features[0].numpy().reshape(features[0].numpy().size),'r')
# plt.subplot(7,1,3)
# plt.title('output of relu')
# plt.plot(X_2_1d)
# #plt.plot(features[1].numpy().reshape(features[1].numpy().size),'r')
# plt.subplot(7,1,4)
# plt.title('output of max')
# plt.plot(X_3_1d)
# #plt.plot(features[2].numpy().reshape(features[2].numpy().size),'r')
# plt.subplot(7,1,5)
# plt.title('output of flatten')
# plt.plot(X_4_1d)
# #plt.plot(features[3].numpy().reshape(features[3].numpy().size),'r')
# plt.subplot(7,1,6)
# plt.title('output of dense 1')
# plt.plot(X_5_1d)
# #plt.plot(features[4].numpy().reshape(features[4].numpy().size),'r')
# plt.subplot(7,1,7)
# plt.title('output of last layer')
# plt.plot(X_6_1d)
# #plt.plot(features[5].numpy().reshape(features[5].numpy().size),'r')
print('break')

# net here needs to change to the way pytorch outputs the logit; below works only for keras
confusing_input_tensor = torch.from_numpy(confusing_output[None, :, :, :]).float().to('cuda:0')

print('Confusing input Tensor:   ')
print(confusing_input_tensor)


output_probabilities, output_value = net.forward(confusing_input_tensor)
output_value = output_value.squeeze(-1).tolist()
pols_soft = F.softmax(output_probabilities.double(), dim=-1).squeeze(0)
pols_soft /= pols_soft.sum()
pols_soft = pols_soft.tolist()

print('output_probabilities = ' , pols_soft)
print('output_value = ' , output_value)


# #####################################################################
# ################### Plotting images
# #####################################################################
print("")

f = plt.figure()
plt.title('output probability distribution')
plt.stem(pols_soft)
f.savefig('lp/stem.png')
# plt.figure()
# # plt.subplot(1,2,1)
# plt.title('Confusing Input')
# plt.imshow(X_0_1d.reshape(28,28),cmap='gray',vmin=0, vmax=1)
# plt.colorbar()
# plt.axis('off')
#plt.autoscale(enable=False)

# plt.subplot(1,2,2)
# plt.title('Original')
# plt.imshow(X_test[1].reshape(28,28),cmap='gray',vmin=0, vmax=1)
# plt.colorbar()
# plt.axis('off')
"""


plt.figure()
plt.subplot(4,1,1)
plt.title('Confusing')
plt.imshow(X_0_1d.reshape(28,28),cmap='gray',vmin=0, vmax=1)
plt.colorbar()
plt.axis('off')
#plt.autoscale(enable=False)


plt.subplot(4,1,2)
plt.title('1-D input')
plt.plot(X_0_1d.reshape(784))


plt.subplot(4,1,3)
plt.title('output probability distribution')
plt.stem(output_probabilities)


plt.subplot(4,1,4)
plt.title('output classification variable from the LP')
plt.stem(X_15_1d)

"""
print('break')