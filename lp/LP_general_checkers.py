import cplex
import numpy as np
import random

import torch
import torch.nn.functional as F
from copy import deepcopy
from scipy.spatial.distance import jensenshannon
DEBUG = True


# combined
def get_confusing_input_pool(net, orig_env, config_file, debug=True, value=True, policy=True, board_size=8,
                                 value_ceil=0, mu=0.5, device='cuda:0', writer=None,lp_states_in_pool=5): #TODO change to 5
    """

    """
    DEVICE = device

    net = deepcopy(net)

    layers = [(name, param.size()) for name, param in net.named_parameters()]

    if debug:
        print(layers)
        print(len(layers))
        print('env_config and net have been imported')
        print('break')

    #################################################################################### build variable names
    X_dictionary_names = {}

    for layer in layers:
        if debug:
            print(layer)
        l = list(layer[1])
        if len(l) == 1:
            X_dictionary_names[("X_{0}".format(layer[0]))] = tuple(layer[1])
        else:
            X_dictionary_names[("X_{0}".format(layer[0]))] = tuple(layer[1][1:])

    #
    # add X_input with shape input_shape at the beginning of X_dictionary_names if the net does not have input Layer !!

    names_string = list(X_dictionary_names.keys())

    if debug:
        print('names_string = ', names_string)
        print('X_dictionary_names = ', X_dictionary_names)

    weight_dims = list(X_dictionary_names.values())

    # get params from pytorch
    net_params = list(net.parameters())

    #shape = net.get_output_shapes(policy=True, value=True, num_actions=256)

    #TODO: did some hardcoding here so that we could run on all GPUs on machines
    shape = [[8, 8, 4], [7, 7, 32], [7, 7, 32], [6, 6, 64],
             [6, 6, 64], 2304, 256, 128, 128, 128, 128, 128,
             256, 128, 128, 128, 1]
    #TODO: uncomment 2 lines below for more robust
    #shape = net.get_output_shapes(policy=True, value=True)
    #shape[-5] = 256
    print(shape)
    # 6 by 6 mazenet should match these shapes.
    # shape = ((6, 6, 5), (5, 5, 32), (5, 5, 32), (4, 4, 64), (4, 4, 64),
    # (256), (256), (128), (128), (128), (128), (128), (1), (1))

    if debug:
        print(f'shape = {shape}')
        #print(f'weight_dims = {weight_dims}')

    ################################ BUILD MANUALLY VARIABLE NAMES:
    # shape = ((6, 6, 5), (5, 5, 32), (5, 5, 32), (4, 4, 64), (4, 4, 64),
    # (1024), (256), (128), (128), (128), (128), (128), (6))

    # need the following for both heads :
    # shape = ((6, 6, 5), (5, 5, 32), (5, 5, 32), (4, 4, 64), (4, 4, 64),
    #           (1024), (256), (128), (128), (128), (128), (128), (6),
    #                                                               (128), (128), (1), (1))


    # input into Conv1
    X_0 = {(i, j, k): 'X_0(i{0},j{1},k{2})'.format(i, j, k) for (i, j, k) in
           build_indicies_dictionary([shape[0][0], shape[0][1], shape[0][2]])}
    # conv1 --> relu
    X_1 = {(i, j, k): 'X_1(i{0},j{1},k{2})'.format(i, j, k) for (i, j, k) in
           build_indicies_dictionary([shape[1][0], shape[1][1], shape[1][2]])}
    # relu --> conv2
    X_2 = {(i, j, k): 'X_2(i{0},j{1},k{2})'.format(i, j, k) for (i, j, k) in
           build_indicies_dictionary([shape[2][0], shape[2][1], shape[2][2]])}
    # conv2 --> relu
    X_3 = {(i, j, k): 'X_3(i{0},j{1},k{2})'.format(i, j, k) for (i, j, k) in
           build_indicies_dictionary([shape[3][0], shape[3][1], shape[3][2]])}
    # relu --> Flatten
    X_4 = {(i, j, k): 'X_4(i{0},j{1},k{2})'.format(i, j, k) for (i, j, k) in
           build_indicies_dictionary([shape[4][0], shape[4][1], shape[4][2]])}
    # flatten --> dense
    X_5 = {(i): 'X_5(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[5]])}
    # dense --> relu
    X_6 = {(i): 'X_6(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[6]])}
    # relu --> dense
    X_7 = {(i): 'X_7(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[7]])}
    # dense --> relu
    X_8 = {(i): 'X_8(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[8]])}

    # relu --> policy x_10
    #      --> value x_13
    X_9 = {(i): 'X_9(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[9]])}

    if policy:
        # policy head
        # dense --> RELU
        X_10 = {(i): 'X_10(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[10]])}
        X_11 = {(i): 'X_11(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[11]])}
        X_12 = {(i): 'X_12(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[12]])}

    if value:

        # VALUE HEAD
        # (from X_9) dense --> relu
        X_13 = {(i): 'X_13(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[13]])}
        # relu --> Dense
        X_14 = {(i): 'X_14(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[14]])}
        # dense --> VALUE
        X_15 = {(i): 'X_15(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[16]])}




    #TODO HARD CODE SHAPE


    # BINARY



    #################################################################### start cplex:
    problem = cplex.Cplex()

    ############################################## define whether maximize or minimize
    problem.objective.set_sense(problem.objective.sense.minimize)

    ############################################### add variables with bounds (X_input and output of each layer):
    ################################################################
    # this defines BOUNDS and add variables to the cplex problem

    problem.variables.add(names=list(X_0.values()), lb=[0.0] * len(X_0), ub=[1.0] * len(X_0))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in X_0.values()])

     # ub from 1 to 2

    problem.variables.add(names=list(X_1.values()), lb=[-2.0] * len(X_1), ub=[2.0] * len(X_1))
    problem.variables.add(names=list(X_2.values()), lb=[0.0] * len(X_2), ub=[2.0] * len(X_2))

    problem.variables.add(names=list(X_3.values()), lb=[-2.0] * len(X_3), ub=[2.0] * len(X_3))
    problem.variables.add(names=list(X_4.values()), lb=[0.0] * len(X_4), ub=[2.0] * len(X_4))
    problem.variables.add(names=list(X_5.values()), lb=[0.0] * len(X_5), ub=[2.0] * len(X_5))

    problem.variables.add(names=list(X_6.values()), lb=[-2.0] * len(X_6), ub=[2.0] * len(X_6))

    problem.variables.add(names=list(X_7.values()), lb=[0.0] * len(X_7), ub=[2.0] * len(X_7))

    problem.variables.add(names=list(X_8.values()), lb=[-2.0] * len(X_8), ub=[2.0] * len(X_8))

    problem.variables.add(names=list(X_9.values()), lb=[0.0] * len(X_9), ub=[2.0] * len(X_9))

    if policy:
        problem.variables.add(names=list(X_10.values()), lb=[-2.0] * len(X_10), ub=[2.0] * len(X_10))

        problem.variables.add(names=list(X_11.values()), lb=[0.0] * len(X_11), ub=[2.0] * len(X_11))

        problem.variables.add(names=list(X_12.values()), lb=[-10.0] * len(X_12), ub=[10.0] * len(X_12))
    if value:
        problem.variables.add(names=list(X_13.values()), lb=[-5.0] * len(X_13), ub=[5.0] * len(X_13))

        problem.variables.add(names=list(X_14.values()), lb=[0.0] * len(X_14), ub=[5.0] * len(X_14))
        problem.variables.add(names=list(X_15.values()), lb=[-10.0] * len(X_15), ub=[10.0] * len(X_15))
        #problem.variables.add(names=list(X_16.values()), lb=[0.0] * len(X_16), ub=[10.0] * len(X_16))


    ####################################################################### OBJECTIVES

    ### all relus
    #
    #
    problem.objective.set_linear(list(zip(list(X_2.values()), [1.0] * len(X_2))))
    problem.objective.set_linear(list(zip(list(X_4.values()), [1.0] * len(X_4))))
    problem.objective.set_linear(list(zip(list(X_7.values()), [1.0] * len(X_7))))
    problem.objective.set_linear(list(zip(list(X_9.values()), [1.0] * len(X_9))))

    if policy:
        problem.objective.set_linear(list(zip(list(X_11.values()), [1.0] * len(X_11))))

    if value:
        problem.objective.set_linear(list(zip(list(X_14.values()), [1.0] * len(X_14))))
        # minimize linear layer
        problem.objective.set_linear(list(zip(list(X_15.values()), [1.0] * len(X_15))))
    

    #####################################################  CONSTRAINTS:

    #############################################################################################################################
    #############################################     conv                    ###################################################
    #############################################################################################################################


    #####################################################  CONSTRAINTS:

    #############################################################################################################################
    #############################################     conv                    ###################################################
    #############################################################################################################################

    
    X_out = X_1  # this is the output of the (conv) layer
    X_in = X_0  # this is the input to the conv layer
    lay = 0
    # size(input) /= size(output) in the case of a conv layer
    shape_out = shape[1]
    shape_in = shape[0]
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
        # W_nth = W_conv_arr[:, :, :,nnn]
        # print(W_nth)
        # print('n = ', nnn, 'W_nth shape = ', W_nth.shape)
        W_nth = W_nth.reshape(pool_size_W, pool_size_H, pool_size_D)
        # print('n = ', nnn, 'W_nth shape = ', W_nth.shape)

        # print("X_in is ", X_in)
        # print("X_out is ", X_out)
        # for every i,j \in I_out X J_out
        for i in range((shape_out[0])):
            for j in range((shape_out[1])):
                # get the portion of input that will be multiplied
                input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
                input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]

                # do the output
                lin_expr_vars_lhs = [X_out[(i, j, nnn)]]
                lin_expr_vals_lhs = [1.0]

                # print("INPUT I ", input_i)
                # print("INPUT J ", input_j)
                # print("^^^^^^^^^^^^^^^^^^^^^")

                # b_conv_arr[nnn]
                # logger
                # print('output indicies: ',i,j,nnn,'filter number = ',nnn,'sum of weights = ',np.sum(W_nth),' bias: ',b_conv_arr[nnn],' input indicies: ', range(input_i[0][0], input_i[0][1] + 1), range(input_j[0][0], input_j[0][1] + 1))
                # loop to do the summation
                for iii in range(input_i[0][0], input_i[0][1] + 1):
                    for jjj in range(input_j[0][0], input_j[0][1] + 1):
                        for kkk in range(pool_size_D):
                            # print((iii, jjj, kkk))

                            if True is False:
                                if (iii, jjj, kkk) in X_in:
                                    lin_expr_vars_lhs.append(X_in[(iii, jjj, kkk)])
                                else:
                                    continue

                            lin_expr_vars_lhs.append(X_in[(iii, jjj, kkk)])

                            a = round(W_nth[iii - input_i[0][0], jjj - input_j[0][0], kkk].item(), 4)
                            lin_expr_vals_lhs.append(-a)
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(b_conv_arr[nnn].item(), 4)],
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
    shape_in = shape[2]
    # get weights and biases
    #
    # below needs to be pulled out from the pytorch model (torch here)
    # below needs to be pulled out from the pytorch model (torch here)
    # W_conv_arr = model.layers[lay].get_weights()[0]
    W_conv_arr = net_params[2]
    b_conv_arr = net_params[3]
    # W_conv_arr = np.ones(shape=(1,1,32,64))
    # b_conv_arr = np.ones(shape=(64,1))
    # get conv filter parameters:
    shape_W = W_conv_arr.shape
    # print(shape_W)
    # print(f'X_in is {X_in}')

    # get conv filters parameters:
    # strides = model.layers[lay].strides[0]
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
        # print(W_nth.shape)
        # print('n = ', nnn, 'W_nth shape = ', W_nth.shape)
        W_nth = W_nth.reshape(pool_size_W, pool_size_H, pool_size_D)

        # for every i,j \in I_out X J_out
        for i in range((shape_out[0])):
            for j in range((shape_out[1])):
                # get the portion of input that will be multiplied
                input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
                input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]
                # print("INPUT I ", input_i)
                # print("INPUT J ", input_j)
                # print("^^^^^^^^^^^^^^^^^^^^^")

                # do the output
                lin_expr_vars_lhs = [X_out[(i, j, nnn)]]
                lin_expr_vals_lhs = [1.0]

                # b_conv_arr[nnn]

                # logger
                # print('output indicies: ',i,j,nnn,'filter number = ',nnn,'sum of weights = ',np.sum(W_nth),' bias: ',b_conv_arr[nnn],' input indicies: ', range(input_i[0][0], input_i[0][1] + 1), range(input_j[0][0], input_j[0][1] + 1))
                # loop to do the summation
                for iii in range(input_i[0][0], input_i[0][1] + 1):
                    for jjj in range(input_j[0][0], input_j[0][1] + 1):
                        for kkk in range(pool_size_D):
                            lin_expr_vars_lhs.append(X_in[(iii, jjj, kkk)])
                            a = round(W_nth[iii - input_i[0][0], jjj - input_j[0][0], kkk].item(), 4)
                            lin_expr_vals_lhs.append(-a)
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(b_conv_arr[nnn].item(), 4)],
                    names=["(conv_1)_"])

    #############################################################################################################################
    #############################################RELU                         ###################################################
    #############################################################################################################################
    """CONSTRAINTS (ReLU_1_1) """
    #  X_2 >= X_1
    X_out = X_2  # this is the output of the Relu
    X_in = X_1  # this is the input to the Relu
    shape_ = shape[1]
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            for k in range(shape_[2]):
                lin_expr_vars_lhs = [X_out[(i, j, k)]]
                lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

                lin_expr_vars_rhs = [X_in[(i, j, k)]]
                lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs,
                                               val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                    senses=['G'],
                    rhs=[0.0],
                    names=["(ReLU_1_1)_"])
    """CONSTRAINTS (ReLU_1_2) """
    #  X_2 >= 0
    X_out = X_2  # this is the output of the Relu
    X_in = X_1  # this is the input to the Relu
    shape_ = shape[1]
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            for k in range(shape_[2]):
                lin_expr_vars_lhs = [X_out[(i, j, k)]]
                lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

                # lin_expr_vars_rhs = [X_in[(i, j, k)]]
                # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['G'],
                    rhs=[0.0],
                    names=["(ReLU_1_2)_"])

    #############################################################################################################################

    """CONSTRAINTS (ReLU_2_1) """
    #  X_2 >= X_1
    X_out = X_4  # this is the output of the Relu
    X_in = X_3  # this is the input to the Relu
    shape_ = shape[4]
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            for k in range(shape_[2]):
                lin_expr_vars_lhs = [X_out[(i, j, k)]]
                lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

                lin_expr_vars_rhs = [X_in[(i, j, k)]]
                lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs,
                                               val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                    senses=['G'],
                    rhs=[0.0],
                    names=["(ReLU_2_1)_"])
    """CONSTRAINTS (ReLU_2_2) """
    #  X_2 >= 0
    X_out = X_4  # this is the output of the Relu
    # X_in  = X_1 # this is the input to the Relu
    shape_ = shape[4]
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            for k in range(shape_[2]):
                lin_expr_vars_lhs = [X_out[(i, j, k)]]
                lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['G'],
                    rhs=[0.0],
                    names=["(ReLU_2_2)_"])

    #############################################################################################################################

    """CONSTRAINTS (ReLU_3_1) """
    #  X_2 >= X_1
    X_out = X_7  # this is the output of the Relu
    X_in = X_6  # this is the input to the Relu
    shape_ = shape[7]
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
            names=["(ReLU_3_1)_"])
    """CONSTRAINTS (ReLU_3_2) """
    #  X_2 >= 0
    X_out = X_7  # this is the output of the Relu
    # X_in  = X_1 # this is the input to the Relu
    shape_ = shape[7]
    for i in range(shape_):
        lin_expr_vars_lhs = [X_out[(i)]]
        lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

        # lin_expr_vars_rhs = [X_in[(i, j, k)]]
        # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
            senses=['G'],
            rhs=[0.0],
            names=["(ReLU_3_2)_"])

    #############################################################################################################################

    """CONSTRAINTS (ReLU_4_1) """
    #  X_2 >= X_1
    X_out = X_9  # this is the output of the Relu
    X_in = X_8  # this is the input to the Relu
    shape_ = shape[9]
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
            names=["(ReLU_4_1)_"])

    """CONSTRAINTS (ReLU_4_2) """
    #  X_2 >= 0
    X_out = X_9  # this is the output of the Relu
    # X_in  = X_1 # this is the input to the Relu
    shape_ = shape[9]
    for i in range(shape_):
        lin_expr_vars_lhs = [X_out[(i)]]
        lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
            senses=['G'],
            rhs=[0.0],
            names=["(ReLU_4_2)_"])

    #############################################################################################################################

    if policy:

        # CONSTRAINTS (ReLU_5_1)
        #  X_2 >= X_1
        X_out = X_11  # this is the output of the Relu
        X_in = X_10  # this is the input to the Relu
        shape_ = shape[11]
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
                names=["(ReLU_5_1)_"])

        # CONSTRAINTS (ReLU_5_2)
        #  X_2 >= 0
        X_out = X_11  # this is the output of the Relu
        # X_in  = X_1 # this is the input to the Relu
        shape_ = shape[11]
        for i in range(shape_):
            lin_expr_vars_lhs = [X_out[(i)]]
            lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['G'],
                rhs=[0.0],
                names=["(ReLU_5_2)_"])

    #############################################################################################################################

    if value:
        # CONSTRAINTS (ReLU_6_1)
        #  X_2 >= X_1
        X_out = X_14  # this is the output of the Relu
        X_in = X_13  # this is the input to the Relu
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

        # CONSTRAINTS (ReLU_6_2)
        #  X_2 >= 0
        X_out = X_14  # this is the output of the Relu
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

    """CONSTRAINTS (den_1)"""

    # we need X_in, X_out, shape_in, shape_out, weights, and biases

    W_dense_arr = net_params[4]
    # (torch here)
    # W_dense_arr = np.ones(shape=(6400,256))
    # W_dense_arr.reshape((256, 1024))
    b_dense_arr = net_params[5]
    # (torch here)
    # b_dense_arr = np.ones(shape=(256))
    # make the biases an array
    X_out = X_6  # this is the output of the FC layer
    X_in = X_5

    shape_ = shape[6]  # shape of the output of the FC  layer
    shape_in = shape[5]  # shape of the input  of the FC  layer

    # looping over i (length of output)
    for i in range(shape_):

        lin_expr_vars_lhs = [X_out[(i)]]
        lin_expr_vals_lhs = [1.0]
        WW = W_dense_arr[i, :]

        # this loop is for the dot product (shape of input)
        for j in range(shape_in):
            lin_expr_vars_lhs.append(X_in[(j)])
            a = round(-WW[j].item(), 4)
            lin_expr_vals_lhs.append(a)

        bb = b_dense_arr[i]

        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
            senses=['E'],
            rhs=[round(bb.item(), 4)],
            names=["(den_1)_"])

    #############################################################################################################################

    """CONSTRAINTS (den_2)"""

    # we need X_in, X_out, shape_in, shape_out, weights, and biases

    W_dense_arr = net_params[6]
    b_dense_arr = net_params[7]  # make the biases an array

    # W_dense_arr = net.layers[4].get_weights()[0]
    # (torch here)
    # W_dense_arr = np.ones(shape=(256,128))

    # b_dense_arr = net.layers[4].get_weights()[1]
    # (torch here)
    # b_dense_arr = np.ones(shape=(128))

    X_out = X_8  # this is the output of the FC layer
    X_in = X_7

    shape_ = shape[8]  # shape of the output of the FC  layer
    shape_in = shape[7]  # shape of the input  of the FC  layer

    # looping over i (length of output)
    for i in range(shape_):

        lin_expr_vars_lhs = [X_out[(i)]]
        lin_expr_vals_lhs = [1.0]
        WW = W_dense_arr[i, :]

        # this loop is for the dot product (shape of input)
        for j in range(shape_in):
            lin_expr_vars_lhs.append(X_in[(j)])
            a = round(-WW[j].item(), 4)
            lin_expr_vals_lhs.append(a)

        bb = b_dense_arr[i]

        problem.linear_constraints.add(
            lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
            senses=['E'],
            rhs=[round(bb.item(), 4)],
            names=["(den_2)_"])

    #############################################################################################################################
    if policy:
        """CONSTRAINTS (den_3)"""

        # we need X_in, X_out, shape_in, shape_out, weights, and biases

        W_dense_arr = net_params[8]
        b_dense_arr = net_params[9]
        # (torch here)
        # W_dense_arr = np.ones(shape=(128,128))

        # b_dense_arr = net.layers[4].get_weights()[1]
        # (torch here)
        # b_dense_arr = np.ones(shape=(128))

        X_out = X_10  # this is the output of the FC layer into POLICY HEAD
        X_in = X_9

        shape_ = shape[10]  # shape of the output of the FC  layer
        shape_in = shape[9]  # shape of the input  of the FC  layer

        # looping over i (length of output)
        for i in range(shape_):

            lin_expr_vars_lhs = [X_out[(i)]]
            lin_expr_vals_lhs = [1.0]
            WW = W_dense_arr[i, :]

            # this loop is for the dot product (shape of input)
            for j in range(shape_in):
                lin_expr_vars_lhs.append(X_in[(j)])
                a = round(-WW[j].item(), 4)
                lin_expr_vals_lhs.append(a)

            bb = b_dense_arr[i]

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[round(bb.item(), 4)],
                names=["(den_3)_"])

    if value:
        """CONSTRAINTS (den_3)"""

        # we need X_in, X_out, shape_in, shape_out, weights, and biases

        W_dense_arr = net_params[12]
        b_dense_arr = net_params[13]
        # (torch here)
        # W_dense_arr = np.ones(shape=(128,128))

        # b_dense_arr = net.layers[4].get_weights()[1]
        # (torch here)
        # b_dense_arr = np.ones(shape=(128))

        X_out = X_13  # this is the output of the FC layer into VALUE HEAD
        X_in = X_9

        shape_ = shape[13]  # shape of the output of the FC  layer
        shape_in = shape[9]  # shape of the input  of the FC  layer

        # looping over i (length of output)
        for i in range(shape_):

            lin_expr_vars_lhs = [X_out[(i)]]
            lin_expr_vals_lhs = [1.0]
            WW = W_dense_arr[i, :]

            # this loop is for the dot product (shape of input)
            for j in range(shape_in):
                lin_expr_vars_lhs.append(X_in[(j)])
                a = round(-WW[j].item(), 4)
                lin_expr_vals_lhs.append(a)

            bb = b_dense_arr[i]

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[round(bb.item(), 4)],
                names=["(den_3)_"])
    #############################################################################################################################

    #############################################################################################################################

    #############################################################################################################################
    ##################################################### flatten   ##################################################
    #############################################################################################################################
    if policy and False:
        # CONSTRAINTS (den_4)
        # LAST IN POLICY HEAD
        # we need X_in, X_out, shape_in, shape_out, weights, and biases

        # (torch here)
        # W_dense_arr = net.layers[4].get_weights()[0]
        # b_dense_arr = net.layers[4].get_weights()[1]         # make the biases an array

        W_dense_arr = net_params[10]
        # (torch here)
        # W_dense_arr = np.ones(shape=(128,128))

        b_dense_arr = net_params[11]
        # (torch here)
        # b_dense_arr = np.ones(shape=(128))

        X_out = X_12  # this is the output of the FC layer
        X_in = X_11

        shape_ = shape[12]  # shape of the output of the FC  layer
        shape_in = shape[11]  # shape of the input  of the FC  layer

        # looping over i (length of output)
        for i in range(shape_):

            lin_expr_vars_lhs = [X_out[(i)]]
            lin_expr_vals_lhs = [1.0]
            WW = W_dense_arr[i, :]

            # this loop is for the dot product (shape of input)
            for j in range(shape_in):
                lin_expr_vars_lhs.append(X_in[(j)])
                a = round(-WW[j].item(), 4)
                lin_expr_vals_lhs.append(a)

            bb = b_dense_arr[i]

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[round(bb.item(), 4)],
                names=["(den_4)_"])

    """CONSTRAINTS (Fltt)"""
    # X_5 = flatten(X_4)

    X_out = X_5  # this is the output of the Flatten
    X_in = X_4  # this is the input to the Flatten
    shape_ = shape[5]  # shape of the output of the flatten  layer
    shape_in = shape[4]  # shape of the input  of the flatten  layer
    # ini
    l = 0
    for i in range(shape_in[0]):
        for j in range(shape_in[1]):
            for k in range(shape_in[2]):
                lin_expr_vars_lhs = [X_in[(i, j, k)]]
                lin_expr_vals_lhs = [1.0]
                lin_expr_vars_rhs = [X_out[(l)]]
                lin_expr_vals_rhs = [-1.0]
                l = l + 1
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs,
                                               val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                    senses=['E'],
                    rhs=[0.0],
                    names=["(Fltt)_"])

    # constraints v and vi
    ######################################################################################################################
    if policy and False:

        """CONSTRAINTS (FINAL POLICY DENSE)"""

        # we need X_in, X_out, shape_in, shape_out, weights, and biases

        W_dense_arr = net_params[10]  # make the weights an array
        b_dense_arr = net_params[11]

        X_out = X_12
        X_in = X_11

        shape_ = shape[12]  # shape of the output of the FC  layer
        shape_in = shape[11]  # shape of the input  of the FC  layer

        # looping over i (length of output)
        for i in range(shape_):

            lin_expr_vars_lhs = [X_out[(i)]]
            lin_expr_vals_lhs = [1.0]
            WW = W_dense_arr[i, :]

            # this loop is for the dot product (shape of input)
            for j in range(shape_in):
                lin_expr_vars_lhs.append(X_in[(j)])
                a = round(-WW[j].item(), 4)
                lin_expr_vals_lhs.append(a)

            bb = b_dense_arr[i]

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['E'],
                rhs=[round(bb.item(), 4)],
                names=[f"(den_final_value{i}"])

        # CONSTRAINTS (v)
        # we need X_in, X_out, shape_in, shape_out, weights, and biases, and number of classes
        number_of_classes = 256
        lin_expr_vars_lhs = []
        lin_expr_vals_lhs = []
        X_temp_1 = X_12

        # ADDED FOR v
        # mu = -0.5

        for i in range(number_of_classes):

            lin_expr_vars_lhs = [X_temp_1[(i)]]
            # TODO: hardcode 5/6
            lin_expr_vals_lhs = [(number_of_classes - 1) / number_of_classes]
            # temp_set = np.setdiff1d([1, 2, 3, 4, 5, 6], i)
            for j in range(number_of_classes):
                if j == i:
                    continue
                lin_expr_vars_lhs.append(X_temp_1[(j)])

                aa = 1 / number_of_classes
                a = round(aa, 4)
                lin_expr_vals_lhs.append(a)

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['G'],
                rhs=[-mu],
                names=[f"(v)_{i}"])
            #print('CONSTRAINT v - last dense with softmax -  is added')

        # CONSTRAINTS (vi)
        # we need X_in, X_out, shape_in, shape_out, weights, and biases, and number of classes
        # torch here

        X_temp_1 = X_12

        # ADDED FOR vi
        # radius
        # DEFINED IN FUNCTION PARAMS
        # mu = 0.5

        for i in range(number_of_classes):

            lin_expr_vars_lhs = [X_temp_1[(i)]]
            lin_expr_vals_lhs = [1.0]

            for j in range(number_of_classes):
                if i == j:
                    continue
                lin_expr_vars_lhs.append(X_temp_1[(j)])

                aa = 1 / number_of_classes
                a = round(aa, 4)
                lin_expr_vals_lhs.append(a)

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['L'],
                rhs=[mu],
                names=[f"(vi)_{i}"])

            #print('CONSTRAINT vi - last dense with softmax -  is added')

        print('break')
        


    #############################################################################################################################

    #############################################################################################################################
    ##################################################### constraint BINARY !!!!   ##################################################
    #############################################################################################################################


    #############################################################################################################################

    """CONSTRAINTS (Constraint_ch5 constant zeros"""

    """CONSTRAINT BINARY (men + kings) DOES NOT EXCEED 12 for player0"""
    # All ones to help neural network find board edges in padded convolutions
    X_in = X_0
    # 8, 8, 4
    shape_out = shape[0]
    lin_expr_vars_lhs = list()
    lin_expr_vals_lhs = list()

    for i in range(shape_out[0]):
        for j in range(shape_out[1]):
            lin_expr_vars_lhs.append(X_in[(i, j, 0)])
            lin_expr_vals_lhs.append(1.0)
            lin_expr_vars_lhs.append(X_in[(i, j, 1)])
            lin_expr_vals_lhs.append(1.0)

    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['L'],
        rhs=[12.0],
        names=[f"(Constraint_binary_menAndKings0_{i}_{j}"])
    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['G'],
        rhs=[1.0],
        names=[f"(Constraint_binary_menAndKings_atleastOne_{i}_{j}"])

    """CONSTRAINT BINARY (men + kings) DOES NOT EXCEED 12 for player1"""
    # All ones to help neural network find board edges in padded convolutions
    X_in = X_0
    # 8, 8, 4
    shape_out = shape[0]
    lin_expr_vars_lhs = list()
    lin_expr_vals_lhs = list()

    for i in range(shape_out[0]):
        for j in range(shape_out[1]):
            lin_expr_vars_lhs.append(X_in[(i, j, 2)])
            lin_expr_vals_lhs.append(1.0)
            lin_expr_vars_lhs.append(X_in[(i, j, 3)])
            lin_expr_vals_lhs.append(1.0)

    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['L'],
        rhs=[12.0],
        names=[f"(Constraint_binary_menAndKings1_{i}_{j}"])
    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['G'],
        rhs=[1.0],
        names=[f"(Constraint_binary_menAndKings1_atleastOne_{i}_{j}"])

    ##########################################################################

    """CONSTRAINT BINARY WHITE SPACES"""
    # All ones to help neural network find board edges in padded convolutions
    X_in = X_0
    # 8, 8, 4
    shape_out = shape[0]
    lin_expr_vars_lhs = list()
    lin_expr_vals_lhs = list()

    for i in range(shape_out[0]):

        for j in range(shape_out[1]):
            # modified empty space in top left corner; nonempty in bottom left corner
            if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                lin_expr_vars_lhs.append(X_in[(i, j, 0)])
                lin_expr_vals_lhs.append(1.0)

                lin_expr_vars_lhs.append(X_in[(i, j, 1)])
                lin_expr_vals_lhs.append(1.0)

                lin_expr_vars_lhs.append(X_in[(i, j, 2)])
                lin_expr_vals_lhs.append(1.0)

                lin_expr_vars_lhs.append(X_in[(i, j, 3)])
                lin_expr_vals_lhs.append(1.0)

    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['E'],
        rhs=[0.0],
        names=[f"(Constraint_binary_whiteSpaces"])

    """CONSTRAINT BINARY NO OVERLAP"""
    # All ones to help neural network find board edges in padded convolutions
    X_in = X_0
    # 8, 8, 4
    shape_out = shape[0]

    for i in range(shape_out[0]):

        for j in range(shape_out[1]):
            lin_expr_vars_lhs = list()
            lin_expr_vals_lhs = list()
            for k in range(shape_out[2]):
                lin_expr_vars_lhs.append(X_in[(i, j, k)])
                lin_expr_vals_lhs.append(1.0)

            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                senses=['L'],
                rhs=[1.0],
                names=[f"(Constraint_binary_NoOverlap"])

    """CONSTRAINT BINARY NO men in back row"""
    # All ones to help neural network find board edges in padded convolutions
    X_in = X_0
    # 8, 8, 4
    shape_out = shape[0]
    lin_expr_vars_lhs = []
    lin_expr_vals_lhs = []
    for j in range(8):
        lin_expr_vars_lhs.append(X_in[(7, j, 0)])
        lin_expr_vals_lhs.append(1)

    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['E'],
        rhs=[0.0],
        names=[f"(Constraint_binary_backrow0"])

    """CONSTRAINT BINARY NO men in back row2"""
    # All ones to help neural network find board edges in padded convolutions
    X_in = X_0
    # 8, 8, 4
    shape_out = shape[0]
    lin_expr_vars_lhs = []
    lin_expr_vals_lhs = []
    for j in range(8):
        lin_expr_vars_lhs.append(X_in[(0, j, 2)])
        lin_expr_vals_lhs.append(1)

    problem.linear_constraints.add(
        lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
        senses=['E'],
        rhs=[0.0],
        names=[f"(Constraint_binary_backrow1"])

    #### try to print either i,j,k mode or only certaint contraints, bounds, or only objectives
    # problem.write( filename='MNIST_digits_.lp')

    '''### this is only used for MIP (Mixed Integer Programming)
    problem.parameters.mip.tolerances.integrality.set(1e-4)
    # problem.parameters.mip.tolerances.mipgap.set(0.01)
    # problem.parameters.mip.tolerances.absmipgap.set(0.01)
    problem.parameters.mip.tolerances.mipgap.set(1e-4)
    problem.parameters.mip.tolerances.absmipgap.set(1e-4)'''

    problem.parameters.mip.pool.intensity.set(1)
    problem.parameters.mip.tolerances.mipgap.set(1e-4)
    problem.parameters.mip.tolerances.absmipgap.set(1e-4)
    problem.parameters.mip.tolerances.integrality.set(1e-4)
    problem.parameters.mip.limits.treememory.set(500)

    # should be 5 by default
    problem.parameters.mip.limits.populate.set(lp_states_in_pool)
    postfix = "combined"

    problem.write(filename=f'constraint_check_2{postfix}.lp')

    random.seed()


    problem.parameters.randomseed.set(random.randint(0, 999999))



    problem.populate_solution_pool()

    #problem.solve()

    solutionstatus = problem.solution.status[problem.solution.get_status()]
    print('LP STATUS: ', solutionstatus)
    print("Solution value  = ", problem.solution.get_objective_value())

    # initialize numpy array of zeros to which we map our confusing output dictionary
    confusing_output = np.zeros(shape=(20, board_size, board_size))

    # pulling up the generated input image from the LP
    temp = {k: problem.solution.get_values(id) for (k, id) in X_0.items()}
    print("PRINTING X_0")

    # manual reshaping
    for (key, value) in temp.items():
        confusing_output[key[2], key[0], key[1]] = value

        if key[2] == 1:
            pass
            # print(f'key: {key} ... value: {value}')

    print(confusing_output)
    num_sols_in_pool = problem.solution.pool.get_num()

    confusing_input_tensors = []

    print(f'NUM SOLUTIONS IN THE SOLUTIONS POOL: {num_sols_in_pool}')

    for idx in range(num_sols_in_pool):
        #print(f'SOLUTION{idx}')
        confusing_output = np.zeros(shape=(shape[0][2], shape[0][0], shape[0][1]))
                # pulling up the generated input image from the LP
        temp = {k: problem.solution.pool.get_values(idx, id) for (k, id) in X_0.items()}
        #print("PRINTING X_0")

        # manual reshaping
        for (key, value) in temp.items():
            confusing_output[key[2], key[0], key[1]] = value

            if key[2] == 1:
                pass
                # print(f'key: {key} ... value: {value}')
        confusing_input_tensors.append(torch.from_numpy(confusing_output[None, :, :, :]).float().to(DEVICE))

    temp_cit = [v1 for i, v1 in enumerate(confusing_input_tensors) if not any(torch.equal(v1, v2)
                                                                              for v2 in confusing_input_tensors[:i])]

    confusing_input_tensors = temp_cit


    print('Confusing input Tensors:   ')
    print('length:', len(confusing_input_tensors))

    output_valueS_list = list()
    jh_list = list()
    for confusing_input_tensor in confusing_input_tensors:
        this_state = orig_env.env.env.env.load_state_from_tensor(confusing_input_tensor)

        orig_env.render()

        print(confusing_input_tensor)
        logit_values, output_value = net.forward(confusing_input_tensor)
        output_valueS = torch.tanh(output_value)
        ov_temp = output_valueS
        output_valueS = output_valueS.squeeze(-1).tolist()
        pols_soft = F.softmax(logit_values.double(), dim=-1).squeeze(0)
        pols_soft /= pols_soft.sum()
        pols_soft = pols_soft.tolist()

        jh = jensenshannon(pols_soft, [1/256] * 256)

        #print('logit_values = ', logit_values)
        print('output_value = ', output_value)
        # BEFORE SOFTMAX
        # 0.018, 0.0623, -0.0818, -0.049, 0.0078, 0.003
        #print('output_probabilities = ', pols_soft)
        print('output_value = ', output_valueS)

        jh_list.append(jh)
        output_valueS_list.append(ov_temp.squeeze(-1).tolist())

    return confusing_input_tensors, jh_list, output_valueS_list


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

