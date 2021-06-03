import cplex
import numpy as np
import matplotlib.pyplot as plt
import json5 as json
import torch
import torch.nn.functional as F


DEBUG = True

DEVICE = 'cuda:0'


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




#state = get_confusing_input(net, orig_env, policy_det=False, value_det=True)

#for thing in state:
#    print(thing)