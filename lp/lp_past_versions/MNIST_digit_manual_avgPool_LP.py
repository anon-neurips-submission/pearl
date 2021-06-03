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
import tensorflow as tf
from keras.datasets import mnist
from keras.datasets import mnist
from keras.datasets import mnist
# to calculate accuracy
#from sklearn.metrics import accuracy_score

# loading the dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path="/home/user/Downloads/mnist.npz")


# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


model = load_model("MNIST_digits__avgPool_model.h5")
#results = model.evaluate(X_test, Y_test)
#print("test loss, test acc:", results)

model.summary()


##extractor function:
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

# let the samoe of interest be:
X = X_test[1]
features = extractor(X.reshape(1,28,28,1))


# ###########################################################################################################################
# ######################### conv equation test ##############################################################################
# ###########################################################################################################################
#
# # we want to compare features[0] vs. x_out = conv(x_in)
#
# #feat = features[2].numpy().reshape(shape=model.layers[2].output_shape)
#
# # implementing the summation for convolution to make sure our constraint is good mother fuckers
#
# # get weights and biases"
# W = model.layers[0].get_weights()[0]
# b = model.layers[0].get_weights()[1]
#
# strides = model.layers[0].strides[0]
#
# pool_size_W = W.shape[0]
# pool_size_H = W.shape[1]
# pool_size_D = W.shape[2]
#
#
#
# # i and j here is in the shape of the output
# #input_i = [(i * strides, (i * strides) + pool_size_W - 1)]
# #input_j = [(j * strides, (j * strides) + pool_size_H - 1)]
#
# #X_input_image = features[1].numpy().reshape(12,12,32)
# X_input_image = X
#
#
# X_out_shape =model.layers[0].output_shape[1:]
# X_out         = np.zeros(shape=(X_out_shape))
#
# weights_shape = W.shape[0:-1]
#
#
# number_of_filters = W.shape[-1]
#
# # for every filter in the conv layer
# for n in range((number_of_filters)):
#     # for every i,j \in I_out X J_out
#     for i in range((X_out_shape[0])):
#         for j in range((X_out_shape[1])):
#             #for k in range((X_out_shape[2])):
#
#             # get the portion of input that will be multiplied
#             input_i = [(i * (strides), (i * (strides)) + pool_size_W - 1)]
#             input_j = [(j * (strides), (j * (strides)) + pool_size_H - 1)]
#             #input_k = [(0,32)]
#             # get the nth filter
#             W_nth = W[:,:,:,n].reshape(pool_size_W,pool_size_H,pool_size_D)
#             #W_nth = np.rot90(W_nth, 2)
#             # point by point multiplicatoins
#
#             X_input_image_portion = X_input_image[input_i[0][0]:input_i[0][1]+1      ,     input_j[0][0]:input_j[0][1]+1  ,
#                                      0:pool_size_D].reshape(pool_size_W,pool_size_H,pool_size_D)
#
#             print('indicies: ', range(input_i[0][0], input_i[0][1] + 1), range(input_j[0][0], input_j[0][1] + 1))
#
#             # X_input_image_portion = X_input_image[input_i[0][0]:input_i[0][1]+1      ,     input_j[0][0]:input_j[0][1]+1  ,
#             #                         0:pool_size_D]
#
#
#             temp = np.multiply(W_nth , X_input_image_portion,dtype='float32')
#             X_out[i,j,n] = np.sum(temp,dtype='float32') + b[n]
#
#
#             # implement Relu
#             #if X_out[i,j,n] < 0:
#             #    X_out[i,j,n] = 0.0
#
#
#
#
#
# # GOAL: compare features[0] vs. x_out = conv(x_in)
#
# v_1 = X_out.reshape(X_out.size)
#
# v_2 = features[0].numpy().reshape(X_out.size)
#
#
# plt.figure()
# plt.subplot(3,1,1)
# plt.title('both')
# plt.plot(v_1)
# plt.plot(v_2,'r')
#
# plt.subplot(3,1,2)
# plt.title('conv manually')
# plt.plot(v_1)
#
# plt.subplot(3,1,3)
# plt.title('conv using keras')
# plt.plot(v_2,'r')
#
#
# ######################################################################




print('break')


############ number of classes
number_of_classes = model.layers[-1].output_shape[1]
print('number of classes = ' , number_of_classes)

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
X_dictionary_names={}
for layer in model.layers:
   X_dictionary_names[("X_{0}".format(layer.name))] =  layer.output_shape[1:]

# add X_input with shape input_shape at the beginning of X_dictionary_names if the model does not have input Layer !!
input_shape = model.layers[0].input_shape[1:]
new_element = {'X_input': input_shape}
X_dictionary_names = {**new_element,**X_dictionary_names}

#print('X_dictionary_names = ' , X_dictionary_names)

names_string = list(X_dictionary_names.keys())
#names_string = names_string[2:] #this is only for VGG test model 3
print('names_string = ' , names_string)

shape  = list(X_dictionary_names.values())

################################ BUILD MANUALLY VARIABLE NAMES:

X_0 = {(i,j,k): 'X_0(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[0][0],shape[0][1], shape[0][2]])}
X_1 = {(i,j,k): 'X_1(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[1][0],shape[1][1], shape[1][2]])}
X_2 = {(i,j,k): 'X_2(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[2][0],shape[2][1], shape[2][2]])}
X_3 = {(i,j,k): 'X_3(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape[3][0],shape[3][1], shape[3][2]])}
X_4 = {(i): 'X_4(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[4][0]])}
X_5 = {(i): 'X_5(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[5][0]])}
X_6 = {(i): 'X_6(i{0})'.format(i) for (i) in build_indicies_dictionary([shape[6][0]])}


# # binary variable for relu
# shape_  = shape[1]
# A_1 = {(i,j,k): 'A_1(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape_[0],shape_[1], shape_[2]])}

# # binary variable for maxPooling
# shape_  = shape[2]
# A_2 = {(i,j,k): 'A_2(i{0},j{1},k{2})'.format(i,j,k) for (i,j,k) in build_indicies_dictionary([shape_[0],shape_[1], shape_[2]])}



#all_vars_ids = list(X_0.values()) + list(X_1.values())

#################################################################### start cplex:
problem = cplex.Cplex()

############################################## define whether maximize or minimize
problem.objective.set_sense(problem.objective.sense.minimize)

############################################### add variables with bounds (X_input and output of each layer):
################################################################ this defines BOUNDS and add variables to the cplex problem



problem.variables.add(names=list(X_0.values()), lb=[0.0] * len(X_0), ub=[1.0] * len(X_0))
#problem.variables.add(names=list(X_1.values()))
problem.variables.add(names=list(X_1.values()),lb=[-5.0] * len(X_1), ub=[5.0] * len(X_1))
problem.variables.add(names=list(X_2.values()),lb=[   0.0] * len(X_2), ub=[5.0] * len(X_2))
problem.variables.add(names=list(X_3.values()),lb=[   0.0] * len(X_3), ub=[5.0] * len(X_3))
problem.variables.add(names=list(X_4.values()),lb=[   0.0] * len(X_4), ub=[5.0] * len(X_4))
problem.variables.add(names=list(X_5.values()),lb=[-5.0] * len(X_5), ub=[5.0] * len(X_5))
problem.variables.add(names=list(X_6.values()),lb=[-20.0] * len(X_6), ub=[20.0] * len(X_6))


######### MAKE THE NEWLY ADDED VARIABLE FOR RELU BINARY
# problem.variables.add(names=list(A_1.values()), lb=[0.0] * len(A_1), ub=[1.0] * len(A_1))
# problem.variables.set_types([(i, problem.variables.type.binary) for i in A_1.values()])

# problem.variables.add(names=list(A_2.values()), lb=[0.0] * len(A_2), ub=[1.0] * len(A_2))
# problem.variables.set_types([(i, problem.variables.type.binary) for i in A_2.values()])

# don't know what below does, but Andre in his code is using this
#problem.variables.advanced.protect([i for i in all_vars_ids])

####################################################################### OBJECTIVES

### testing only:
#problem.objective.set_linear(list(zip(list(X_0.values()), [1.0] * len(X_0))))



# ### relu summation: X_2 [remove in case we are using the binary variables]
problem.objective.set_linear(list(zip(list(X_2.values()), [1.0] * len(X_2))))

# ### max pooling: X_3
#problem.objective.set_linear(list(zip(list(X_3.values()), [1.0] * len(X_3))))

# ### last dense X_6
#problem.objective.set_linear(list(zip(list(X_6.values()), [1.0] * len(X_6))))



#target_label = 9
#problem.objective.set_linear(list(zip(list([X_6[(target_label)]]), [-1.0])))





#
# WWW = model.layers[-2].get_weights()[0]
#
# # all i\in [M]\t
# array_of_classes = np.asarray(range(number_of_classes))
# array_of_classes_other_than_target = np.delete(array_of_classes,[target_label])
#
#
# WWW_tar = WWW[:,target_label]
#
# # to get vector VV \in [1024]
#
# VV = [None]*len(WWW_tar)
#
# for i in range(len(WWW_tar)):
#     temp = -WWW_tar[i] + np.sum(WWW[i,array_of_classes_other_than_target])
#     VV[i] = round(temp.item(),4)*1.0
#
# problem.objective.set_linear(list(zip(list(X_5.values()), VV)))
# print('archetypal constraint is added')








#####################################################  CONSTRAINTS:

"""CONSTRAINTS (ia_1) """
# this is for X_1 = conv(X_0)

print('constraint (ia) is added')
X_out = X_1  # this is the output of the (conv) layer
X_in = X_0  # this is the input to the conv layer
lay = 0
# size(input) /= size(output) in the case of a conv layer
shape_out = shape[1]
shape_in  = shape[0]
# get weights and biases
#

W_conv_arr = model.layers[lay].get_weights()[0]
b_conv_arr = model.layers[lay].get_weights()[1]
# get conv filter parameters:
shape_W = W_conv_arr.shape

# get conv filters parameters:
strides = model.layers[lay].strides[0]
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
                names=["(ia_1)_"])


#
##################################################################### relu:

"""CONSTRAINTS (ic_1) """
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
                                names=["(ic_1)_"])



"""CONSTRAINTS (iii_AvgPool)"""
# this is for X_3x = maxPool(X_2)
print('CONSTRAINT iii - AVG pooling -  is added')
X_out = X_3 # this is the output of the Max
X_in  = X_2 # this is the input to the  Max
shape_   = shape[3] # shape of the output of the max pool  layer
shape_in = shape[2] # shape of the input  of the max pool  layer
lay = 2
# for max pooling, the shape of the channel in input and output is the same

# get strides and pool size and loop accordingly to build constraints
Pool_size = model.layers[lay].pool_size
Pool_size_W = Pool_size[0]
Pool_size_H = Pool_size[1]
strides = model.layers[lay].strides[0]

# loop over channels:
for k in range(shape_in[2]):
    # loop over output indicies
    for i in range(shape_[0]):
        for j in range(shape_[1]):
            # build the corresponding input indices from i, j, Pool_size, and strides
                input_i = [( i * strides , (i * strides) + Pool_size_W -   1)]
                input_j = [( j * strides , (j * strides) + Pool_size_W -   1)]
                lin_expr_vars_lhs = [X_out[(i, j, k)]]
                lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)
                # loop over above indicies to build to contraints
                for iii in range(input_i[0][0], input_i[0][1] + 1):
                    for jjj in range(input_j[0][0], input_j[0][1] + 1):
                #        lin_expr_vars_lhs = [X_out[(i, j, k)]]


                        lin_expr_vars_lhs.append(X_in[(iii, jjj, k)])
                        lin_expr_vals_lhs.append(-0.25)

                problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(lin_expr_vars_lhs ,
                                                       val=lin_expr_vals_lhs) ],
                            senses=['E'],
                            rhs=[0.0],
                            names=["(iii_AvgPool)_"])




"""CONSTRAINTS (Fltt)"""

X_out = X_4 # this is the output of the Flatten
X_in  = X_3 # this is the input to the Flatten
shape = list(X_dictionary_names.values())
shape_   = shape[4] # shape of the output of the flatten  layer
shape_in = shape[3] # shape of the input  of the flatten  layer

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



"""CONSTRAINTS (iv_1)"""
W_dense_arr = model.layers[4].get_weights()[0]
b_dense_arr = model.layers[4].get_weights()[1]         # make the biases an array
X_out = X_5     # this is the output of the FC layer
X_in  = X_4

shape_   = shape[5] # shape of the output of the FC  layer
shape_in = shape[4  ] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_[0]):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

    # this loop is for the dot product (shape of input)
    for j in range(shape_in[0]):

        lin_expr_vars_lhs.append(X_in[(j)])
        a = round(-WW[j].item(),4)
        lin_expr_vals_lhs.append(a)


#
    bb = b_dense_arr[i]

    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(bb.item(),4)],
                    names=["(iv)_"])

#
"""CONSTRAINTS (iv_2)"""
W_dense_arr = model.layers[5].get_weights()[0]
b_dense_arr = model.layers[5].get_weights()[1]         # make the biases an array
X_out = X_6     # this is the output of the FC layer
X_in  = X_5

shape_   = shape[6] # shape of the output of the FC  layer
shape_in = shape[5] # shape of the input  of the FC  layer

# looping over i (length of output)
for i in range(shape_[0]):

    lin_expr_vars_lhs = [X_out[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW = W_dense_arr[:,i]

    # this loop is for the dot product (shape of input)
    for j in range(shape_in[0]):

        lin_expr_vars_lhs.append(X_in[(j)])
        a = round(-WW[j].item(),4)
        lin_expr_vals_lhs.append(a)


#
    bb = b_dense_arr[i]

    problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
                    senses=['E'],
                    rhs=[round(bb.item(),4)],
                    names=["(iv)_"])


# """CONSTRAINTS (target)"""
# # make the X_6[target_lbl] > X_6[i], for all i\in [M]\target_lbl
# array_of_classes = np.asarray(range(number_of_classes))
# array_of_classes_other_than_target = np.delete(array_of_classes,[target_label])
#
# for lbl in array_of_classes_other_than_target:
#     print("loop",lbl, "works")
#
#     lin_expr_vars_tar = [X_6[(target_label)]]
#     lin_expr_vals_tar = [1.0]
#     lin_expr_vars_lbl = [X_6[(lbl)]]
#     lin_expr_vals_lbl = [-1.0]
#
#     problem.linear_constraints.add(
#                     lin_expr=[cplex.SparsePair( lin_expr_vars_tar+lin_expr_vars_lbl, val=lin_expr_vals_tar+lin_expr_vals_lbl)],
#                     senses=['G'],
#                     rhs=[1.0],
#                     names=["(target)_"])


# """CONSTRAINTS (confusing)"""
# # make the X_6[target_lbl] > X_6[i], for all i\in [M]\target_lbl
# any_lbl = 0
# array_of_classes = np.asarray(range(number_of_classes))
# array_of_classes_other_than_target = np.delete(array_of_classes,[any_lbl])
#
# for lbl in array_of_classes_other_than_target:
#     print("loop",lbl, "works")
#
#     lin_expr_vars_tar = [X_6[(any_lbl)]]
#     lin_expr_vals_tar = [1.0]
#     lin_expr_vars_lbl = [X_6[(lbl)]]
#     lin_expr_vals_lbl = [-1.0]
#
#     problem.linear_constraints.add(
#                     lin_expr=[cplex.SparsePair( lin_expr_vars_tar+lin_expr_vars_lbl, val=lin_expr_vals_tar+lin_expr_vals_lbl)],
#                     senses=['E'],
#                     rhs=[0.0],
#                     names=["(confusing)_"])

print('break')

"""CONSTRAINTS (v)"""

W_dense_arr = model.layers[5].get_weights()[0]         # make the weights an array
bias        = model.layers[5].get_weights()[1]
X_temp_1 = X_6
X_temp_0 = X_5

vector_V = np.sum(W_dense_arr,axis=1)
print('CONSTRAINT v - last dense with softmax -  is added')
for i in range(number_of_classes):

    lin_expr_vars_lhs = [X_temp_1[(i)]]
    lin_expr_vals_lhs = [1.0]
    WW  = W_dense_arr[:,i]


    # this
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
#
#
#
#
#
"""CONSTRAINTS (vi)"""
W_dense_arr = model.layers[5].get_weights()[0]         # make the weights an array
bias        = model.layers[5].get_weights()[1]         # make the weights an array
X_temp_1 = X_6
X_temp_0 = X_5

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


## hard code the input for debugging purposes
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



# """CONSTRAINTS (\sum out_prob = 1)"""
# # the sum of the output of the last layer is 1
#
# # define the variable:
# X_out_prob = X_6
#
# # loop over number of classes (length of the output vector)
# lin_expr_vars_lhs = []
# lin_expr_vals_lhs = []
# for i in range(number_of_classes):
#     lin_expr_vars_lhs.append(X_out_prob[(i)])
#     lin_expr_vals_lhs.append(1.0)
#
# problem.linear_constraints.add(
#                             lin_expr=[cplex.SparsePair( lin_expr_vars_lhs, val=lin_expr_vals_lhs)],
#                             senses=['E'],
#                             rhs=[1.0],
#                             names=["(\sum out_prob = 1)_"])


#### try to print either i,j,k mode or only certaint contraints, bounds, or only objectives
#problem.write( filename='MNIST_digits_AvgPooling_constraint.lp')






# problem.parameters.mip.tolerances.integrality.set(1e-2)
# problem.parameters.mip.tolerances.mipgap.set(0.1)
# problem.parameters.mip.tolerances.absmipgap.set(0.1)


problem.solve()



solutionstatus = problem.solution.status[problem.solution.get_status()]
print('LP STATUS: ' , solutionstatus)
print("Solution value  = ", problem.solution.get_objective_value())


temp = {k: problem.solution.get_values(id) for (k, id) in X_0.items()}
X_0_1d = np.array(list(temp.values()))
print('Solution of ', 'X_0', ' = ', temp)

temp = {k: problem.solution.get_values(id) for (k, id) in X_1.items()}
X_1_1d = np.array(list(temp.values()))
print('Solution of ', 'X_1', ' = ', temp)

temp = {k: problem.solution.get_values(id) for (k, id) in X_2.items()}
X_2_1d = np.array(list(temp.values()))
print('Solution of ', 'X_2', ' = ', temp)

temp = {k: problem.solution.get_values(id) for (k, id) in X_3.items()}
X_3_1d = np.array(list(temp.values()))
print('Solution of ', 'X_3', ' = ', temp)

temp = {k: problem.solution.get_values(id) for (k, id) in X_4.items()}
X_4_1d = np.array(list(temp.values()))
print('Solution of ', 'X_4', ' = ', temp)

temp = {k: problem.solution.get_values(id) for (k, id) in X_5.items()}
X_5_1d = np.array(list(temp.values()))
print('Solution of ', 'X_5', ' = ', temp)

temp = {k: problem.solution.get_values(id) for (k, id) in X_6.items()}
X_6_1d = np.array(list(temp.values()))
print('Solution of ', 'X_6', ' = ', temp)


# temp = {k: problem.solution.get_values(id) for (k, id) in A_1.items()}
# A_6_1d = np.array(list(temp.values()))
# print('Solution of ', 'A_1', ' = ', temp)

plt.figure()
plt.subplot(7,1,1)
plt.title('input 1D')
plt.plot(X_0_1d)
plt.plot(X.reshape(28*28),'r')
plt.subplot(7,1,2)
plt.title('output of conv')
plt.plot(X_1_1d)
plt.plot(features[0].numpy().reshape(features[0].numpy().size),'r')
plt.subplot(7,1,3)
plt.title('output of relu')
plt.plot(X_2_1d)
#plt.plot(features[1].numpy().reshape(features[1].numpy().size),'r')
plt.subplot(7,1,4)
plt.title('output of max')
plt.plot(X_3_1d)
#plt.plot(features[2].numpy().reshape(features[2].numpy().size),'r')
plt.subplot(7,1,5)
plt.title('output of flatten')
plt.plot(X_4_1d)
plt.plot(features[3].numpy().reshape(features[3].numpy().size),'r')
plt.subplot(7,1,6)
plt.title('output of dense 1')
plt.plot(X_5_1d)
plt.plot(features[4].numpy().reshape(features[4].numpy().size),'r')
plt.subplot(7,1,7)
plt.title('output of last layer')
plt.plot(X_6_1d)
plt.plot(features[6].numpy().reshape(features[6].numpy().size),'r')


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
plt.stem(X_6_1d)


print('break')