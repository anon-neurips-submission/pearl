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


def get_confusing_input(net, orig_env, config_file, policy_det=True, value_det=False, debug=False):
    """

    """
    if policy_det == value_det:
        return ValueError("pick either policy equalization or value minimization")
    with open(config_file) as f:
        config = json.load(f)

    layers = [(name, param.size()) for name, param in net.named_parameters()]


    if debug:
        print(layers)
        print(len(layers))
        print('env_config and net have been imported')
        print('break')

    #################################################################################### build variable names
    X_dictionary_names={}

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
        print('names_string = ' , names_string)
        print('X_dictionary_names = ' , X_dictionary_names)


    weight_dims  = list(X_dictionary_names.values())

    # get params from pytorch
    net_params = list(net.parameters())


    shape = net.get_output_shapes(policy=policy_det, value=value_det)


    # 6 by 6 mazenet should match these shapes.
    #shape = ((6, 6, 5), (5, 5, 32), (5, 5, 32), (4, 4, 64), (4, 4, 64),
    # (256), (256), (128), (128), (128), (128), (128), (1), (1))

    if debug:
        print(f'shape = {shape}')
        print(f'weight_dims = {weight_dims}')

    ################################ BUILD MANUALLY VARIABLE NAMES:
    # shape = ((6, 6, 5), (5, 5, 32), (5, 5, 32), (4, 4, 64), (4, 4, 64),
    # (1024), (256), (128), (128), (128), (128), (128), (6))
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



    Agent_pos = {(i,j): 'Agent_pos(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}

    A = {(i,j): 'A(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}
    A_fact = {(i,j): 'A_fact(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}
    A_sum = {(i,j): 'A_sum(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}
    Wood_inv = {(i): 'Wood_inv(i{0})'.format(i) for (i) in build_indicies_dictionary([1])}
    Tool_inv = {(i): 'Tool_inv(i{0})'.format(i) for (i) in build_indicies_dictionary([1])}

    A_to_inv = {(i,j): 'A_to_inv(i{0},j{1})'.format(i,j) for (i,j) in build_indicies_dictionary([shape[0][0],shape[0][1]])}

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

    if value_det:
        X_13 = {(i): 'X_13(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[13]])}
        problem.variables.add(names=list(X_13.values()), lb=[0.0] * len(X_13), ub=[10.0] * len(X_13))
        problem.objective.set_linear(list(zip(list(X_13.values()), [1.0] * len(X_13))))

    ######### add the binary variable Agent_pos
    problem.variables.add(names=list(Agent_pos.values()), lb=[0.0] * len(Agent_pos), ub=[1.0] * len(Agent_pos))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in Agent_pos.values()])

    ######### add the binary varible for wood
    problem.variables.add(names=list(A.values()), lb=[0.0] * len(A), ub=[1.0] * len(A))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in A.values()])

    ######### add the binary variable A_factory
    problem.variables.add(names=list(A_fact.values()), lb=[0.0] * len(A_fact), ub=[1.0] * len(A_fact))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in A_fact.values()])

    # TODO: generalize inventory upper bounds here
    ######### add the binary variable Wood inv
    problem.variables.add(names=list(Wood_inv.values()), lb=[0.0] * len(Wood_inv), ub=[1.0] * len(Wood_inv))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in Wood_inv.values()])

    ######### add the binary variable Tool inv
    problem.variables.add(names=list(Tool_inv.values()), lb=[0.0] * len(Tool_inv), ub=[1.0] * len(Tool_inv))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in Tool_inv.values()])

    ######### add the binary variable A_sum
    problem.variables.add(names=list(A_sum.values()), lb=[0.0] * len(A_sum), ub=[1.0] * len(A_sum))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in A_sum.values()])

    ######### add the binary variable A_to_inv
    problem.variables.add(names=list(A_to_inv.values()), lb=[0.0] * len(A_to_inv), ub=[1.0] * len(A_to_inv))
    problem.variables.set_types([(i, problem.variables.type.binary) for i in A_to_inv.values()])

    ####################################################################### OBJECTIVES

    ### all relus
    #
    #
    problem.objective.set_linear(list(zip(list(X_2.values()), [1.0]  * len(X_2 ))))
    problem.objective.set_linear(list(zip(list(X_4.values()), [1.0]  * len(X_4 ))))
    problem.objective.set_linear(list(zip(list(X_7.values()), [1.0]  * len(X_7 ))))
    problem.objective.set_linear(list(zip(list(X_9.values()), [1.0]  * len(X_9 ))))
    problem.objective.set_linear(list(zip(list(X_11.values()), [1.0] * len(X_11))))


    #####################################################  CONSTRAINTS:

    #############################################################################################################################
    #############################################     conv                    ###################################################
    #############################################################################################################################

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
        #print(W_nth.shape)
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
                                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
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


                problem.linear_constraints.add(
                                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
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
                            lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
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


        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair( lin_expr_vars_lhs , val=lin_expr_vals_lhs)],
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

        problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                            senses=['G'],
                            rhs=[0.0],
                            names=["(ReLU_5_2)_"])

    #############################################################################################################################


    if value_det:
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

    if value_det:
        W_dense_arr = net_params[12]
        b_dense_arr = net_params[13]
    else: # if policy_det
        W_dense_arr = net_params[8]
        b_dense_arr = net_params[9]
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



    #CONSTRAINTS (den_4)

    # we need X_in, X_out, shape_in, shape_out, weights, and biases

    # (torch here)
    #W_dense_arr = net.layers[4].get_weights()[0]
    #b_dense_arr = net.layers[4].get_weights()[1]         # make the biases an array

    if value_det:
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
                            names=["(den_4)_"])


    if policy_det:
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

        # constraints v and vi
        ######################################################################################################################

        # CONSTRAINTS (v)
        # we need X_in, X_out, shape_in, shape_out, weights, and biases, and number of classes
        number_of_classes = 6
        # torch here
        W_dense_arr = net_params[10]  # make the weights an array
        bias = net_params[11]

        # W_dense_arr = net.layers[4].get_weights()[0]
        # (torch here)
        # W_dense_arr = np.ones(shape=(128,6))

        # b_dense_arr = net.layers[4].get_weights()[1]
        # (torch here)
        # bias = np.ones(shape=(6))

        X_temp_1 = X_12
        X_temp_0 = X_11

        # vector_V = np.sum(W_dense_arr)
        vector_V = W_dense_arr.sum(axis=0)
        vector_V_1 = W_dense_arr.sum(axis=1)

        print('CONSTRAINT v - last dense with softmax -  is added')
        for i in range(number_of_classes):

            lin_expr_vars_lhs = [X_temp_1[(i)]]
            lin_expr_vals_lhs = [1.0]
            WW = W_dense_arr[i, :]

            for j in range(len(WW)):
                lin_expr_vars_lhs.append(X_temp_0[(j)])

                ##### vals has to be np.float not np.float32
                aa = (-WW[j] + (1 / number_of_classes) * vector_V[j])
                a = round(aa.item(), 4)
                lin_expr_vals_lhs.append(a)
            bias_good = round(bias[i].item(), 4)
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['G'],
                rhs=[0.0],
                names=["(v)_"])
            print('CONSTRAINT v - last dense with softmax -  is added')

        # CONSTRAINTS (vi)
        # we need X_in, X_out, shape_in, shape_out, weights, and biases, and number of classes
        number_of_classes = 6
        # torch here
        # torch here
        # W_dense_arr = net.layers[5].get_weights()[0]         # make the weights an array
        # bias        = net.layers[5].get_weights()[1]

        # W_dense_arr = net.layers[4].get_weights()[0]
        # (torch here)
        # W_dense_arr = np.ones(shape=(128,6))
        W_dense_arr = net_params[10]
        b_dense_arr = net_params[11]
        # b_dense_arr = net.layers[4].get_weights()[1]
        # (torch here)
        # bias = np.ones(shape=(6))
        X_temp_1 = X_12
        X_temp_0 = X_11

        vector_V = W_dense_arr.sum(axis=0)

        print('CONSTRAINT v - last dense with softmax -  is added')
        for i in range(number_of_classes):

            lin_expr_vars_lhs = [X_temp_1[(i)]]
            lin_expr_vals_lhs = [1.0]
            WW = W_dense_arr[i, :]

            # this
            for j in range(len(WW)):
                lin_expr_vars_lhs.append(X_temp_0[(j)])

                ##### vals has to be np.float not np.float32
                aa = (WW[j] - (1 / number_of_classes) * vector_V[j])
                a = round(aa.item(), 4)
                lin_expr_vals_lhs.append(a)
            bias_good = round(bias[i].item(), 4)
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['G'],
                rhs=[0.0],
                names=["(vi)_"])
            print('CONSTRAINT vi - last dense with softmax -  is added')


            print('break')

            #########################################################################
            # radius from optimal solution. smaller means longer to find solution
            # RADIUS FROM SOLUTION CONSTRAINTS (Constraint_X_LFC)
            ##########################################################################
            lin_expr_vars_lhs=[]
            lin_expr_vals_lhs=[]

            lin_expr_vars_lhs.append(X_12[(0)])
            lin_expr_vals_lhs.append(1.0)

            # we want that RHS to be less than 0.5 (value)
            problem.linear_constraints.add(
                                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs,  val=lin_expr_vals_lhs)],
                                    senses=['L'],
                                    rhs=[2.0],
                                    names=["(X_LFC_fromObj)_"])
        else:
            print('value head')

            # radius from optimal solution. smaller means longer to find solution
            # CONSTRAINTS (Constraint_X_LFC)
            lin_expr_vars_lhs = []
            lin_expr_vals_lhs = []

            lin_expr_vars_lhs.append(X_13[(0)])
            lin_expr_vals_lhs.append(1.0)

            # we want that RHS to be less than 0.5 (value)
            problem.linear_constraints.add(
                lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                senses=['L'],
                rhs=[0.2],
                names=["(X_LFC_fromObj)_"])

    #############################################################################################################################


    #############################################################################################################################
    ##################################################### constraint BINARY !!!!   ##################################################
    #############################################################################################################################

    # WE HAVE ALREADY DEFINED A BINARY CONSTRAINT A which the same size as the input tensor

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


    st_A_max = config['env']['params']['placements'][0]['random_placements']

    # sum of list equal to NUMBER OF WOOD ON BOARD
    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                    senses=['E'],
                    # should be 6 for example vvvv
                    rhs=[st_A_max],
                    names=["(Constraint_binary_1"])



    """CONSTRAINTS (Constraint_binary_2"""
    # picking some input indicies in which X = A ;
    # for example X_in(0,0,0) and X_in(0,0,1) and X_in(0,0,2) shoould equal A

    # special_input_indicies = [(0,0,0),(0,0,1),(0,0,2)]

    X_out = A
    X_in  = X_0 # this is the input to the Flatten
    # shape_out   = shape[0] # shape of the output of the flatten  layer
    # shape_iin   = shape[0] # shape of the input  of the flatten  layer


    for i in range(0, shape_out[0]):
        for j in range(0, shape_out[1]):

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

    st_B_max = config['env']['params']['placements'][1]['random_placements']
    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(lin_expr_vars_lhs, lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[st_B_max],
                    names=["(Constraint_binary_2_1"])



    """CONSTRAINTS (Constraint_binary_2_2"""
    # picking some input indicies in which X = A ;
    # for example X_in(0,0,0) and X_in(0,0,1) and X_in(0,0,2) shoould equal A

    # special_input_indicies = [(0,0,0),(0,0,1),(0,0,2)]

    X_out = A_fact
    X_in  = X_0 # this is the input to the Flatten
    # shape_out   = shape[0] # shape of the output of the flatten  layer
    # shape_iin   = shape[0] # shape of the input  of the flatten  layer


    for i in range(0, shape_out[0]):
        for j in range(0, shape_out[1]):

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

    # THIS WILL STAY one because SINGLE agent
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


    for i in range(0, shape_out[0]):
        for j in range(0, shape_out[1]):

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
    max_inv_wood = config['env']['params']['inventory'][0]['capacity']

    for i in range(shape_out[0]):

        for j in range(shape_out[1]):
            # make a 2 because max 2 woods
            lin_expr_vars_1 = [X_out[(i, j, 3)]]



            lin_expr_vals_1 = [max_inv_wood * 1.0] * len(lin_expr_vars_1)


            # wood inv slice
            lin_expr_vars_2 = [X_in[(0)]]
            lin_expr_vals_2 = [-1.0] * len(lin_expr_vars_2)
            # sum of list equal to max inv wood


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
    max_tool_inv = config['env']['params']['inventory'][1]['capacity']

    for i in range(shape_out[0]):

        for j in range(shape_out[1]):
            lin_expr_vars_1 = [X_out[(i, j, 4)]]
            lin_expr_vals_1 = [max_tool_inv * 1.0] * len(lin_expr_vars_1)


            # wood inv slice
            lin_expr_vars_2 = [X_in[(0)]]
            lin_expr_vals_2 = [-1.0] * len(lin_expr_vars_2)
            # sum of list equal to 2


            problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(lin_expr_vars_1 + lin_expr_vars_2, lin_expr_vals_1 + lin_expr_vals_2)],
                            senses=['E'],
                            rhs=[0],
                            names=["(Constraint_binary_5_1"])


    """CONSTRAINTS (Constraint_binary_6_1"""
    #\sum(A) = 50 ; only 50 entries of the binary tensor is one,


    shape_out   = shape[0]


    for i in range(shape_out[0]):

        for j in range(shape_out[1]):
            lin_expr_vars_1 = [A[(i, j)]]
            lin_expr_vals_1 = [1.0] * len(lin_expr_vars_1)



            lin_expr_vars_2 = [A_fact[(i, j)]]
            lin_expr_vals_2 = [1.0] * len(lin_expr_vars_2)


            lin_expr_vars_3 = [A_sum[(i, j)]]
            lin_expr_vals_3 = [-1.0] * len(lin_expr_vars_2)
            # sum of list equal to 2


            problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(lin_expr_vars_1 + lin_expr_vars_2 + lin_expr_vars_3, lin_expr_vals_1 + lin_expr_vals_2 + lin_expr_vals_3)],
                            senses=['E'],
                            rhs=[0],
                            names=["(Constraint_binary_6_1"])

    #############################################################################################################################



    #### try to print either i,j,k mode or only certaint contraints, bounds, or only objectives
    #problem.write( filename='MNIST_digits_.lp')

    ### this is only used for MIP (Mixed Integer Programming)
    problem.parameters.mip.tolerances.integrality.set(1e-4)
    #problem.parameters.mip.tolerances.mipgap.set(0.01)
    #problem.parameters.mip.tolerances.absmipgap.set(0.01)
    problem.parameters.mip.tolerances.mipgap.set(1e-4)
    problem.parameters.mip.tolerances.absmipgap.set(1e-4)

    postfix = "policy" if policy_det else "value"

    problem.write( filename=f'constraint_check_{postfix}.lp')
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

    # this is the output the of the last layer from the LP
    temp = {k: problem.solution.get_values(id) for (k, id) in X_12.items()}
    X_15_1d = np.array(list(temp.values()))
    print('Solution of ', 'X_12', ' = ', temp)

    plt.figure()
    plt.subplot(7,1,1)
    plt.title('input 1D')
    plt.plot(X_0_1d)

    confusing_input_tensor = torch.from_numpy(confusing_output[None, :, :, :]).float().to(DEVICE)

    print('Confusing input Tensor:   ')
    print(confusing_input_tensor)

    pos_got, done_got, special_tiles_got, inventory_got = orig_env.load_state_from_tensor(confusing_input_tensor)
    print(pos_got)
    print(done_got)
    print(special_tiles_got)
    print(inventory_got)

    output_probabilities, output_value = net.forward(confusing_input_tensor)
    output_value = output_value.squeeze(-1).tolist()
    pols_soft = F.softmax(output_probabilities.double(), dim=-1).squeeze(0)
    pols_soft /= pols_soft.sum()
    pols_soft = pols_soft.tolist()

    print('output_probabilities = ' , pols_soft)
    print('output_value = ' , output_value)
    net.training = True

    return confusing_input_tensor


#state = get_confusing_input(net, orig_env, policy_det=False, value_det=True)

#for thing in state:
#    print(thing)