# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation
from keras.utils import np_utils
from keras.models import load_model
import cplex
import numpy as np
import keras
import matplotlib.pyplot as plt


# # to calculate accuracy
# #from sklearn.metrics import accuracy_score
#
# # loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # building the input vector from the 28x28 pixels
# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# # normalizing the data to help with the training
# X_train /= 255
# X_test /= 255
#
# # one-hot encoding using keras' numpy-related utilities
# n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
# Y_train = np_utils.to_categorical(y_train, n_classes)
# Y_test = np_utils.to_categorical(y_test, n_classes)
# print("Shape after one-hot encoding: ", Y_train.shape)
#
#
#model = load_model("MNIST_digits__avgPool_dense_softmax_together_model.h5")
#results = model.evaluate(X_test, Y_test)
#print("test loss, test acc:", results)
#
# model.summary()
#
#
# ##extractor function:
# extractor = keras.Model(inputs=model.inputs,
#                         outputs=[layer.output for layer in model.layers])
#
# # let the samoe of interest be:
# X = X_test[1]
# features = extractor(X.reshape(1,28,28,1))







print('break')


############ number of classes
# number_of_classes = model.layers[-1].output_shape[1]
# print('number of classes = ' , number_of_classes)

number_of_classes = 6

#######################################################################

# build variables indicies based on shape
def build_indicies_dictionary(ls):
    dictionary = {}
    if len(ls) < 3:
        for i in range(ls[0]):
            dictionary[(i)] = (i)
    else:
        for i in range(ls[0]):
            for j in range(ls[1]):
                for k in range(ls[2]):
                    dictionary[(i, j, k)] = (i, j, k)
    return dictionary


#XX_1_dictionary = build_indicies_dictionary(N_i,N_j,N_k)

#################################################################################### build variable names
#################################################################################### build variable names
# X_dictionary_names={}
# for layer in model.layers:
#    X_dictionary_names[("X_{0}".format(layer.name))] =  layer.output_shape[1:]
#
# # add X_input with shape input_shape at the beginning of X_dictionary_names if the model does not have input Layer !!
# input_shape = model.layers[0].input_shape[1:]
# new_element = {'X_input': input_shape}
# X_dictionary_names = {**new_element,**X_dictionary_names}
#
# #print('X_dictionary_names = ' , X_dictionary_names)
#
# names_string = list(X_dictionary_names.keys())
# #names_string = names_string[2:] #this is only for VGG test model 3
# print('names_string = ' , names_string)

#shape  = list(X_dictionary_names.values())

# input shape depends on maze_shape which gives the first two and the third is in channels

shape=[(6,6,5),(6,6,32),(6,6,32),(6,6,64),(6,6,64),(6400),(256),(256),(128),(128),(128),(128),(128),(128),(6),(6)]

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
X_13 = {(i): 'X_13(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[13]])}
X_14 = {(i): 'X_14(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[14]])}
X_15 = {(i): 'X_14(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[15]])}



#################################################################### start cplex:
problem = cplex.Cplex()

############################################## define whether maximize or minimize
problem.objective.set_sense(problem.objective.sense.minimize)

############################################### add variables with bounds (X_input and output of each layer):
################################################################ this defines BOUNDS and add variables to the cplex problem


problem.variables.add(names=list(X_0.values()), lb=[0.0] * len(X_0), ub=[1.0] * len(X_0))
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

problem.variables.add(names=list(X_12.values()),lb=[-5.0] * len(X_12), ub=[5.0] * len(X_12))

problem.variables.add(names=list(X_13.values()),lb=[0.0] * len(X_13), ub=[5.0] * len(X_13))

problem.variables.add(names=list(X_14.values()),lb=[-5.0] * len(X_14), ub=[5.0] * len(X_14))

problem.variables.add(names=list(X_15.values()),lb=[0.0] * len(X_15), ub=[1.0] * len(X_15))



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

"""CONSTRAINTS (conv_1) """
# this is for X_1 = conv(X_0)
# we need X_in, X_out, shape_in, shape_out, weights, and biases

X_out = X_1  # this is the output of the (conv) layer
X_in = X_0  # this is the input to the conv layer
lay = 0
# size(input) /= size(output) in the case of a conv layer
shape_out = shape[1]
shape_in  = shape[0]
# get weights and biases
#
# below needs to be pulled out from the pytorch model (torch here)
#W_conv_arr = model.layers[lay].get_weights()[0]
W_conv_arr = np.ones(shape=(1,1,5,32))
b_conv_arr = np.ones(shape=(32,1))
# get conv filter parameters:
shape_W = W_conv_arr.shape

# get conv filters parameters:
strides = 1
pool_size_W = W_conv_arr.shape[0]
pool_size_H = W_conv_arr.shape[1]
pool_size_D = W_conv_arr.shape[2]

number_of_filters = W_conv_arr.shape[-1]

# for every filter in the conv layer
for nnn in range((number_of_filters)):
    # get the nth filter
    W_nth = W_conv_arr[:, :, :, nnn]
    # print('n = ', nnn, 'W_nth shape = ', W_nth.shape)
    W_nth = W_nth.reshape(pool_size_W, pool_size_H, pool_size_D)

    # for every i,j \in I_out X J_out
    for i in range((shape_out[0])):
        for j in range((shape_out[1])):
            # get the portion of input that will be multiplied
            input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
            input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]

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
W_conv_arr = np.ones(shape=(1,1,32,64))
b_conv_arr = np.ones(shape=(64,1))
# get conv filter parameters:
shape_W = W_conv_arr.shape

# get conv filters parameters:
#strides = model.layers[lay].strides[0]
strides = 1
pool_size_W = W_conv_arr.shape[0]
pool_size_H = W_conv_arr.shape[1]
pool_size_D = W_conv_arr.shape[2]

number_of_filters = W_conv_arr.shape[-1]

# for every filter in the conv layer
for nnn in range((number_of_filters)):
    # get the nth filter
    W_nth = W_conv_arr[:, :, :, nnn]
    # print('n = ', nnn, 'W_nth shape = ', W_nth.shape)
    W_nth = W_nth.reshape(pool_size_W, pool_size_H, pool_size_D)

    # for every i,j \in I_out X J_out
    for i in range((shape_out[0])):
        for j in range((shape_out[1])):
            # get the portion of input that will be multiplied
            input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
            input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]

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
#X_in  = X_1 # this is the input to the Relu
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
shape_ = shape[6]
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
shape_ = shape[8]
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


"""CONSTRAINTS (ReLU_5_1) """
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

"""CONSTRAINTS (ReLU_5_2) """
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


"""CONSTRAINTS (ReLU_6_1) """
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

"""CONSTRAINTS (ReLU_6_2) """
#  X_2 >= 0
X_out = X_13  # this is the output of the Relu
# X_in  = X_1 # this is the input to the Relu
shape_ = shape[13]
for i in range(shape_):
    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

    # lin_expr_vars_rhs = [X_in[(i, j, k)]]
    # lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

    problem.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
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


#W_dense_arr = model.layers[4].get_weights()[0]
# (torch here)
W_dense_arr = np.ones(shape=(6400,256))

#b_dense_arr = model.layers[4].get_weights()[1]
# (torch here)
b_dense_arr = np.ones(shape=(256))
# make the biases an array
X_out = X_6     # this is the output of the FC layer
X_in  = X_5

shape_   = shape[6  ] # shape of the output of the FC  layer
shape_in = shape[5  ] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

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


#W_dense_arr = model.layers[4].get_weights()[0]
#b_dense_arr = model.layers[4].get_weights()[1]         # make the biases an array

#W_dense_arr = model.layers[4].get_weights()[0]
# (torch here)
W_dense_arr = np.ones(shape=(256,128))

#b_dense_arr = model.layers[4].get_weights()[1]
# (torch here)
b_dense_arr = np.ones(shape=(128))

X_out = X_8     # this is the output of the FC layer
X_in  = X_7

shape_   = shape[8  ] # shape of the output of the FC  layer
shape_in = shape[7  ] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

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
#W_dense_arr = model.layers[4].get_weights()[0]
#b_dense_arr = model.layers[4].get_weights()[1]         # make the biases an array


#W_dense_arr = model.layers[4].get_weights()[0]
# (torch here)
W_dense_arr = np.ones(shape=(128,128))

#b_dense_arr = model.layers[4].get_weights()[1]
# (torch here)
b_dense_arr = np.ones(shape=(128))


X_out = X_10     # this is the output of the FC layer
X_in  = X_9

shape_   = shape[10] # shape of the output of the FC  layer
shape_in = shape[9] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

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

"""CONSTRAINTS (den_4)"""

# we need X_in, X_out, shape_in, shape_out, weights, and biases

# (torch here)
#W_dense_arr = model.layers[4].get_weights()[0]
#b_dense_arr = model.layers[4].get_weights()[1]         # make the biases an array


#W_dense_arr = model.layers[4].get_weights()[0]
# (torch here)
W_dense_arr = np.ones(shape=(128,128))

#b_dense_arr = model.layers[4].get_weights()[1]
# (torch here)
b_dense_arr = np.ones(shape=(128))

X_out = X_12     # this is the output of the FC layer
X_in  = X_11

shape_   = shape[12] # shape of the output of the FC  layer
shape_in = shape[11] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

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

#############################################################################################################################



#############################################################################################################################
##################################################### flatten   ##################################################
#############################################################################################################################

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

#############################################################################################################################



#############################################################################################################################
##################################################### constraint (v)(vi)   ##################################################
#############################################################################################################################

"""CONSTRAINTS (v)"""
# we need X_in, X_out, shape_in, shape_out, weights, and biases, and number of classes
number_of_classes = 6
# torch here
#W_dense_arr = model.layers[5].get_weights()[0]         # make the weights an array
#bias        = model.layers[5].get_weights()[1]

#W_dense_arr = model.layers[4].get_weights()[0]
# (torch here)
W_dense_arr = np.ones(shape=(128,6))

#b_dense_arr = model.layers[4].get_weights()[1]
# (torch here)
bias = np.ones(shape=(6))


X_temp_1 = X_15
X_temp_0 = X_13


vector_V = np.sum(W_dense_arr,axis=1)
print('CONSTRAINT v - last dense with softmax -  is added')
for i in range(number_of_classes):

    lin_expr_vars_lhs = [X_temp_1[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW  = W_dense_arr[:,i]

    for j in range(len(WW)):

        lin_expr_vars_lhs.append(X_temp_0[(j)])

        ##### vals has to be np.float not np.float32
        aa = (-WW[j]+(1/number_of_classes)*vector_V[j])
        a = round(aa.item(),4)
        lin_expr_vals_lhs.append(a)
    bias_good = round(bias[i].item(), 4)
    problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair( lin_expr_vars_lhs,  val=lin_expr_vals_lhs)],
                            senses=['G'],
                            rhs=[0.0],
                            names=["(v)_"])
    print('CONSTRAINT v - last dense with softmax -  is added')





"""CONSTRAINTS (vi)"""
# we need X_in, X_out, shape_in, shape_out, weights, and biases, and number of classes
number_of_classes = 6
# torch here
# torch here
#W_dense_arr = model.layers[5].get_weights()[0]         # make the weights an array
#bias        = model.layers[5].get_weights()[1]

#W_dense_arr = model.layers[4].get_weights()[0]
# (torch here)
W_dense_arr = np.ones(shape=(128,6))

#b_dense_arr = model.layers[4].get_weights()[1]
# (torch here)
bias = np.ones(shape=(6))
X_temp_1 = X_15
X_temp_0 = X_13


vector_V = np.sum(W_dense_arr,axis=1)
print('CONSTRAINT v - last dense with softmax -  is added')
for i in range(number_of_classes):

    lin_expr_vars_lhs = [X_temp_1[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

    # this
    for j in range(len(WW)):

        lin_expr_vars_lhs.append(X_temp_0[(j)])

        ##### vals has to be np.float not np.float32
        aa = (WW[j]-(1/number_of_classes)*vector_V[j])
        a = round(aa.item(), 4)
        lin_expr_vals_lhs.append(a)
    bias_good = round(bias[i].item(),4)
    problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                            senses=['G'],
                            rhs=[0.0],
                            names=["(vi)_"])
    print('CONSTRAINT vi - last dense with softmax -  is added')

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


"""CONSTRAINTS (Constraint_X_LFC)"""
lin_expr_vars_lhs=[]
lin_expr_vals_lhs=[]
for i in range(number_of_classes):
    lin_expr_vars_lhs.append(X_15[(i)])
    lin_expr_vals_lhs.append(1.0)
problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair( lin_expr_vars_lhs,  val=lin_expr_vals_lhs)],
                        senses=['L'],
                        rhs=[0.1],
                        names=["(X_LFC_fromObj)_"])
#############################################################################################################################



#### try to print either i,j,k mode or only certaint contraints, bounds, or only objectives
#problem.write( filename='MNIST_digits_.lp')



problem.solve()



solutionstatus = problem.solution.status[problem.solution.get_status()]
print('LP STATUS: ' , solutionstatus)
print("Solution value  = ", problem.solution.get_objective_value())


# pulling up the generated input image from the LP
temp = {k: problem.solution.get_values(id) for (k, id) in X_0.items()}
X_0_1d = np.array(list(temp.values()))
print('Solution of ', 'X_0', ' = ', temp)

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
temp = {k: problem.solution.get_values(id) for (k, id) in X_15.items()}
X_15_1d = np.array(list(temp.values()))
print('Solution of ', 'X_15', ' = ', temp)

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


# model here needs to change to the way pytorch outputs the logit; below works only for keras
output_probabilities = model(X_0_1d.reshape(1,28,28,1))[0]
print('output_probabilities = ' , output_probabilities)


# #####################################################################
# ################### Plotting images
# #####################################################################
print("")

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


print('break')