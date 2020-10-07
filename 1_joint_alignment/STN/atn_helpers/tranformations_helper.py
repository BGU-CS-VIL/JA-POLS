#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:32:35 2018

@author: fredman
"""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tf_funcs
import numpy as np
from tensorflow.python.framework import ops
from scipy.linalg import expm_frechet
import cv2
from random import randint, uniform
from STN.atn_helpers.network import resnet
from STN.atn_helpers.spatial_transformer import transformer
from STN.tasks_for_atn.matrix_exp import expm



def transfromation_parameters_regressor(transformations,images,keep_prob,img_sz,weight_stddev,num_channels,
                                        activation_func, num_stn):

    num_of_params = 6
    #params = network1(images, img_sz,keep_prob,activation_func,num_of_params,weight_stddev,num_channels)
    # params = regress_parameters(images,img_sz,keep_prob,activation,num_of_params,weight_stddev,num_channels)
    x_theta, theta, theta_exp = STN(images, weight_stddev, num_of_params, img_sz, num_channels, num_stn)
    # params = regress_parameters_fc(images,img_sz,keep_prob,activation,num_of_params,weight_stddev,num_channels)
    # params = learn_all_params(images,img_sz,keep_prob,activation,weight_stddev,num_channels)
    # params = learn_all_params_conv(images,img_sz,keep_prob,activation,weight_stddev,num_channels)
    # fc_size = 50
    # filters = 32
    # strides = 1
    # training = True
    # data_format = 'channels_last'
    # # params = resnet(images, filters, training, strides, data_format, weight_stddev, fc_size, keep_prob, num_of_params, img_sz, num_channels)
    #
    # d2 = dict()
    # d2['params'] = params
    # orig_transformation = tf.slice(params, [0, 0], [-1, 6])
    # for transformation in transformations:
    #     if transformation in methods:
    #         transform_maps_dict[transformation] = tf.reshape(orig_transformation, [-1, 6])

    return x_theta, theta, theta_exp


# build Spatial Transformer Network
def STN(images, weight_stddev, num_of_params, img_sz, num_channels, num_stn):

    images = tf.reshape(images, shape=[-1, img_sz[0], img_sz[1], num_channels])
    reuse_weights = False
    out_size = (img_sz[0], img_sz[1])

    theta_all = []
    theta_exp_all = []

    input_ = images
    for i in range(num_stn):
        if i > 0:
            reuse_weights = True

        theta = create_locnet(input_, num_of_params, weight_stddev, reuse_weights)
        theta_exp = expm(-theta)  # compute matrix exponential on {-theta}
        input_, d = transformer(input_, theta_exp, out_size)
        #to avoid the sparse indexing warning, comment the next line, and uncomment the one after it.
        input_ = tf.reshape(input_, shape=[-1, img_sz[0], img_sz[1], num_channels])

        theta_all.append(theta)
        theta_exp_all.append(theta_exp)

    output = input_

    theta_all = tf.convert_to_tensor(theta_all)
    theta_exp_all = tf.convert_to_tensor(theta_exp_all)
    # theta_exp_all = tf.Print(theta_exp_all, [theta_exp_all], message="theta_exp_all: ", summarize=100)

    return output, theta_all, theta_exp_all


def create_locnet(images, num_of_params, weight_stddev, reuse_weights):

    def conv2Layer(feat, outDim):
        weight, bias = createVariable([5,5,int(feat.shape[-1]),outDim],stddev=weight_stddev)
        conv = tf.nn.conv2d(feat,weight,strides=[1,1,1,1],padding="VALID")+bias
        return conv

    def linearLayer(feat,outDim,final=False):
        weight,bias = createVariable([int(feat.shape[-1]),outDim],stddev=0.0 if final else weight_stddev)
        fc = tf.matmul(feat,weight)+bias
        return fc

    with tf.variable_scope("geometric", reuse=reuse_weights):
        feat = images
        with tf.variable_scope("conv1", reuse=reuse_weights):
            feat = conv2Layer(feat,32)
            feat = tf.nn.relu(feat)
        with tf.variable_scope("conv2", reuse=reuse_weights):
            feat = conv2Layer(feat,16)
            feat = tf.nn.relu(feat)
            feat = tf.nn.max_pool(feat,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
        with tf.variable_scope("conv2", reuse=reuse_weights):
            feat = conv2Layer(feat,8)
            feat = tf.nn.relu(feat)
            feat = tf.nn.max_pool(feat,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
        # feat = tf.reshape(feat, [opt.batchSize,-1])
        layer_flat,num_features = flatten_layer(feat)
        with tf.variable_scope("fc3", reuse=reuse_weights):
            layer_flat = linearLayer(layer_flat, 48)
            layer_flat = tf.nn.relu(layer_flat)
        with tf.variable_scope("fc4", reuse=reuse_weights):
            layer_flat = linearLayer(layer_flat, num_of_params,final=False)

    return layer_flat

# auxiliary function for creating weight and bias
def createVariable(weightShape, biasShape=None, stddev=None):

    if biasShape is None: biasShape = [weightShape[-1]]

    # weight = tf.get_variable("weight",shape=weightShape,dtype=tf.float32,
    #                                   initializer=tf.random_normal_initializer(stddev=stddev))
    # bias = tf.get_variable("bias",shape=biasShape,dtype=tf.float32,
    #                               initializer=tf.random_normal_initializer(stddev=stddev))

    weight = tf.truncated_normal(weightShape, stddev=stddev)
    weight = tf.Variable(weight)

    bias = tf.truncated_normal(biasShape, stddev=stddev)
    bias = tf.Variable(bias)
    # bias = tf.Variable(tf.constant(0.01,shape=[biasShape]))

    return weight, bias


#this is the stn's localization network
#here we construct it using a 2-layer FC network.
def regress_parameters(x, img_sz, keep_prob, activation_func, num_of_params, weight_stddev, num_channels):
    with tf.variable_scope("atn"):
        #We'll setup the two-layer localisation network to figure out the
        # parameters for an affine transformation of the input
        # Create variables for fully connected layer
        W_fc_loc1 = weight_variable([img_sz[0] * img_sz[1] * num_channels, 20], weight_stddev)
        b_fc_loc1 = bias_variable([20], weight_stddev)

        W_fc_loc2 = weight_variable([20, num_of_params], weight_stddev)
        # Use identity transformation as starting point
        b_fc_loc2 = bias_variable2([1, num_of_params], weight_stddev)

        # Define the two layer localisation network
        h_fc_loc1 = tf.nn.leaky_relu(tf.matmul(x,W_fc_loc1) + b_fc_loc1)  # tf.nn.leaky_relu
        # We can add dropout for regularizing and to reduce overfitting like so:
        h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1,keep_prob)

        # Second layer
        s_fc_loc2 = tf.nn.leaky_relu(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)  # tf.nn.leaky_relu activation_func

        return s_fc_loc2


#this is just another way to learn all 6 params without having to concat the vectors learned.
#can also just concat (like we did for scaling for example).
def learn_all_params(x,img_sz,keep_prob,activation_func,weight_stddev,num_channels):
    #We'll setup the two-layer localisation network to figure out the
    # parameters for an affine transformation of the input
    # Create variables for fully connected layer
    W_fc_loc1 = weight_variable([img_sz[0] * img_sz[1] * num_channels, 20], weight_stddev)
    b_fc_loc1 = bias_variable([20])

    W_fc_loc2 = weight_variable([20, 6], weight_stddev)
    # Use identity transformation as starting point
    # initial = np.array([[1.,0.,0.],[0.,1.,0.]])
    # initial = initial.astype('float32')
    # initial = initial.flatten()
    # b_fc_loc2 = tf.Variable(initial_value=initial,name='b_fc_loc2')
    initial = tf.constant(0.,shape=[1,6])
    b_fc_loc2 = tf.Variable(initial)

    # Define the two layer localisation network
    h_fc_loc1 = activation_func(tf.matmul(x,W_fc_loc1) + b_fc_loc1)
    # We can add dropout for regularizing and to reduce overfitting like so:
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1,keep_prob)
    # Second layer
    h_fc_loc2 = activation_func(tf.matmul(h_fc_loc1_drop,W_fc_loc2) + b_fc_loc2)

    return h_fc_loc2


#this is the stn's localization network
#here we construct it using a 4-layer CNN network.
def learn_all_params_conv(x,img_sz,keep_prob,activation,weight_stddev,num_channels):
    num_of_params = 6

    # Convolutional Layer 1.
    filter_size1 = 5
    num_filters1 = 32

    # Convolutional Layer 2.
    filter_size2 = 5
    num_filters2 = 32

    # Fully-connected layer.
    fc_size = 32

    # The convolutional layers expect x to be encoded as a 4-dim tensor so we have to
    #reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
    input_images = tf.reshape(x,shape=[-1,img_sz[0],img_sz[1],num_channels])

    layer_conv1,weights_conv1 = \
        new_conv_layer(input=input_images,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       weight_stddev=weight_stddev,
                       use_pooling=True)
    # layer_conv1 = tf.Print(layer_conv1,[layer_conv1],message="layer_conv1: ",summarize=100)

    layer_conv2,weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       weight_stddev=weight_stddev,
                       use_pooling=True)
    # layer_conv2 = tf.Print(layer_conv2,[layer_conv2],message="layer_conv2: ",summarize=100)

    #before sending the input to a non-convolutional layer, need to re-flatten it
    #(need to undo what we did in the reshape before the convolutional layer above)
    layer_flat,num_features = flatten_layer(layer_conv2)
    # layer_flat = tf.Print(layer_flat,[layer_flat],message="layer_flat: ",summarize=100)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             weight_stddev=weight_stddev,
                             use_relu=True)
    # layer_fc1 = tf.Print(layer_fc1,[layer_fc1],message="layer_fc1: ",summarize=100)
    layer_fc1_drop = tf.nn.dropout(layer_fc1,keep_prob)
    # layer_fc1_drop = tf.Print(layer_fc1_drop,[layer_fc1_drop],message="layer_fc1_drop: ",summarize=100)

    layer_fc2 = new_fc_layer(input=layer_fc1_drop,
                             num_inputs=fc_size,
                             num_outputs=fc_size,
                             weight_stddev=weight_stddev,
                             use_relu=True)
    # layer_fc2 = tf.Print(layer_fc2,[layer_fc2],message="layer_fc2: ",summarize=100)
    layer_fc2_drop = tf.nn.dropout(layer_fc2,keep_prob)
    # layer_fc2_drop = tf.Print(layer_fc2_drop,[layer_fc2_drop],message="layer_fc2_drop: ",summarize=100)

    layer_fc22 = unit_fc_layer(input=layer_fc2_drop,
                              num_inputs=fc_size,
                              num_outputs=num_of_params,
                              weight_stddev=weight_stddev,
                              use_relu=False)
    # layer_fc22 = tf.Print(layer_fc22,[layer_fc22],message="layer_fc22: ",summarize=100)

    layer_fc22 = tf.Print(layer_fc22,[layer_fc22],message="layer_fc22: ",summarize=100)

    return layer_fc22


def weight_variable(shape,stddev):
    initial = tf.truncated_normal(shape,stddev=stddev)
    return tf.Variable(initial)
    # initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    # return tf.Variable(initializer(shape))



def bias_variable(shape, stddev):
    initial = tf.truncated_normal(shape,stddev=stddev)
    # initial = tf.constant(0.01,shape=shape)
    return tf.Variable(initial)
    # initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    # return tf.Variable(initializer(shape))
    # initial = tf.constant(0.01, shape=shape) #tf.constant(0.1, shape=shape)
    # # initial = tf.constant([1.,0.,0.,0.,1.,0.], shape=shape)
    # return tf.Variable(initial)


def bias_variable2(shape, stddev):
    initial = tf.truncated_normal(shape,stddev=stddev)
    return tf.Variable(initial)
    # initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    # return tf.Variable(initializer(shape))
    # initial = tf.constant(0.01, shape=shape) #tf.constant(0.1, shape=shape)
    # # initial = tf.constant([1.,0.,0.,0.,1.,0.], shape=shape)
    # return tf.Variable(initial)


# should return a scalar (regulator)
def transformation_regularization_SE(theta):
    BB_T = [[0,0,0,0,0,0],[0,0.5,0,-0.5,0,0],[0,0,1,0,0,0],[0,-0.5,0,0.5,0,0],[0,0,0,0,0,0],[0,0,0,0,0,1]] # 6 x 6
    BB_T = tf.cast(BB_T, tf.float32)
    theta_T = tf.transpose(theta) # 6 x batch_sz
    theta_tilda = tf.matmul(BB_T, theta_T) # 6 x batch_sz
    norm_arr = tf.norm(tf.subtract(theta_T, theta_tilda), axis=0) # 1 x  batch_sz
    regularizer = tf.reduce_sum(tf.square(norm_arr)) # scalar
    #regularizer = tf.reduce_sum(tf.abs(norm_arr))  # scalar
    return regularizer


def transformation_regularization_SIMPLE(theta):
    identity_trans = tf.expand_dims(tf.constant([0.,0.,0.,0.,0.,0.]), 1)
    identity_trans = tf.cast(identity_trans, tf.float32)
    theta_T = tf.transpose(theta) # 6 x batch_sz
    norm_arr = tf.norm(tf.subtract(theta_T, identity_trans), axis=0) # 1 x  batch_sz
    regularizer = tf.reduce_sum(tf.square(norm_arr)) # scalar
    regularizer = tf.cast(regularizer, tf.float32)
    #regularizer = tf.reduce_sum(tf.abs(norm_arr))  # scalar
    return regularizer


# should return a scalar (regulator)
def transformation_regularization_VP(theta):
    BB_T = [[0.5,0,0,0,-0.5,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [-0.5,0,0,0,0.5,0], [0,0,0,0,0,1]] # 6 x 6
    BB_T = tf.cast(BB_T, tf.float32)
    theta_T = tf.transpose(theta) # 6 x batch_sz
    theta_tilda = tf.matmul(BB_T, theta_T) # 6 x batch_sz
    norm_arr = tf.norm(tf.subtract(theta_T, theta_tilda), axis=0) # 1 x  batch_sz
    regularizer = tf.reduce_sum(tf.square(norm_arr)) # scalar
    #regularizer = tf.reduce_sum(tf.abs(norm_arr))  # scalar
    return regularizer


#give a diffrent penalty to each type of transformation magnituted
def transformation_regularization_WEIGHTS(affine_maps,regularizations):
    transformations = [*regularizations]
    stable_transformations = 0.
    unit_transformation = tf.constant([0.,0.,0.,0.,0.,0.])

    for transformation in transformations:
        if transformation in affine_maps:
            maps = affine_maps[transformation]
            if transformation == "sh" or transformation == "t" or transformation == "ane":
                # for these transformation matrices we dind't use the matrix exponential, so need to compare the output transformation to the unit matrix
                # for the "ane" case we'll allow flipping in one direction (= negative determinant), so a matrix like [[-1,0,0],[0,1,0]] with a negative determinant, is allowed.
                maps = tf.abs(maps)
                #maps = tf.Print(maps,[maps],message="maps: ",summarize=100)
                stable_transformations += regularizations[transformation] * tf.reduce_sum(tf.square(tf.subtract(tf.reduce_mean(maps,0),unit_transformation)))
            #            elif transformation == "sc": # testign for scaling, change the "scadadel" to "sc"
            #                l2_l1 = l2_l1_func(maps)
            #               stable_transformations += regularizations[transformation]*tf.reduce_sum(tf.reduce_sum(l2_l1, 0))
            else:  # for these we use the exponent, so need to compare the matrix before the exponent to the zero matrix
                stable_transformations += regularizations[transformation] * tf.reduce_sum(tf.reduce_sum(tf.square(maps),1),0)

    return stable_transformations




def matrix_expnential(matrices,batch_size):
    matrices = tf.cast(matrices,tf.float32)
    x_unpacked = tf.unstack(matrices,num=batch_size)  # defaults to axis 0, returns a list of tensors
    processed = []  # this will be the list of processed tensors
    for t in x_unpacked:
        t = tf.cast(t,tf.float64)
        result_tensor = tf.linalg.expm(t)
        processed.append(result_tensor)

    #output = tf.concat(tf.cast(processed, tf.float32), 0)
    return tf.reshape(processed,[-1,3,3])


#Define the matrix exponential gradient for back propagation
def register_gradient():
    if "MatrixExponential" not in ops._gradient_registry._registry:
        #Adding the gradient for the matrix exponential functionrixExponential" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("MatrixExponential")
        def _expm_grad(op,grad):
            # We want the backward-mode gradient (left multiplication).
            # Let X be the NxN input matrix.
            # Let J(X) be the the N^2xN^2 complete Jacobian matrix of expm at X.
            # Let Y be the NxN previous gradient in the backward AD (left multiplication)
            # We have
            # unvec( ( vec(Y)^T . J(X) )^T )
            #   = unvec( J(X)^T . vec(Y) )
            #   = unvec( J(X^T) . vec(Y) )
            # where the last part (if I am not mistaken) holds in the case of the
            # exponential and other matrix power series.
            # It can be seen that this is now the forward-mode derivative
            # (right multiplication) applied to the Jacobian of the transpose.
            grad_func = lambda x,y:expm_frechet(x,y,compute_expm=False)
            return tf.py_func(grad_func,[tf.transpose(op.inputs[0]),grad],tf.float64)



def unit_fc_layer(input,  # The previous layer.
                  num_inputs,  # Num. inputs from prev. layer.
                  num_outputs,  # Num. outputs.
                  weight_stddev,
                  use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = weight_variable(shape=[num_inputs,num_outputs],stddev=weight_stddev)
    # Use identity transformation as starting point
    # initial = np.array([[0.01,0,0],[0,0.01,0]])
    # initial = initial.astype('float32')
    # initial = initial.flatten()
    # biases = tf.Variable(initial_value=initial,name='b_fc_loc2')
    biases = tf.Variable(tf.constant(0.01, shape=[6]))
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input,weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer



def l2_l1_func(
        maps):  #this function takes the square of all values greater than zero, and the absolute of all those smaller than zero
    m1 = tf.maximum(maps,0)
    #    m1 = tf.square(m1)
    m2 = tf.minimum(maps,0)
    m2 = tf.abs(m2)
    #    m2 *= 1e-5
    m2 * 0
    return tf.add(m1,m2)
    return m1


#this is the stn's localization network
#here we construct it using a 3-layer FC network.
def regress_parameters_fc(x,img_sz,keep_prob,activation_func,num_of_params,weight_stddev,num_channels):
    with tf.variable_scope("atn"):
        #We'll setup the two-layer localisation network to figure out the
        # parameters for an affine transformation of the input
        # Create variables for fully connected layer
        W_fc_loc1 = weight_variable([img_sz[0] * img_sz[1] * num_channels,32],weight_stddev)
        b_fc_loc1 = bias_variable([32])

        W_fc_loc2 = weight_variable([32,32],weight_stddev)
        # Use identity transformation as starting point
        b_fc_loc2 = bias_variable([1,32])

        W_fc_loc3 = weight_variable([32,num_of_params],weight_stddev)
        # Use identity transformation as starting point
        b_fc_loc3 = bias_variable2([1,num_of_params])

        # Define the two layer localisation network
        h_fc_loc1 = tf.nn.leaky_relu(tf.matmul(x,W_fc_loc1) + b_fc_loc1)
        # We can add dropout for regularizing and to reduce overfitting like so:
        h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1,keep_prob)

        # Define the two layer localisation network
        h_fc_loc2 = tf.nn.leaky_relu(tf.matmul(h_fc_loc1_drop,W_fc_loc2) + b_fc_loc2)
        # We can add dropout for regularizing and to reduce overfitting like so:
        h_fc_loc2_drop = tf.nn.dropout(h_fc_loc2,keep_prob)

        # Third layer
        s_fc_loc3 = tf.nn.leaky_relu(tf.matmul(h_fc_loc2_drop,W_fc_loc3) + b_fc_loc3)

        return s_fc_loc3


# %%

#this is the stn's localization network
#here we construct it using a 4-layer CNN network.
def regress_parameters_conv(x,input_size,keep_prob,activation_func,num_of_params,weight_stddev,num_channels):
    with tf.variable_scope("atn"):
        # Convolutional Layer 1.
        filter_size1 = 5
        num_filters1 = 32

        # Convolutional Layer 2.
        filter_size2 = 5
        num_filters2 = 32

        # Fully-connected layer.
        fc_size = 32

        # The convolutional layers expect x to be encoded as a 4-dim tensor so we have to
        #reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
        input_images = tf.reshape(x,shape=[-1,input_size,input_size,num_channels])

        layer_conv1,weights_conv1 = \
            new_conv_layer(input=input_images,
                           num_input_channels=num_channels,
                           filter_size=filter_size1,
                           num_filters=num_filters1,
                           weight_stddev=weight_stddev,
                           use_pooling=True)

        layer_conv2,weights_conv2 = \
            new_conv_layer(input=layer_conv1,
                           num_input_channels=num_filters1,
                           filter_size=filter_size2,
                           num_filters=num_filters2,
                           weight_stddev=weight_stddev,
                           use_pooling=True)

        #before sending the input to a non-convolutional layer, need to re-flatten it
        #(need to undo what we did in the reshape before the convolutional layer above)
        layer_flat,num_features = flatten_layer(layer_conv2)

        layer_fc1 = new_fc_layer(input=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=fc_size,
                                 weight_stddev=weight_stddev,
                                 use_relu=True)
        layer_fc1_drop = tf.nn.dropout(layer_fc1,keep_prob)

        layer_fc2 = new_fc_layer(input=layer_fc1_drop,
                                 num_inputs=fc_size,
                                 num_outputs=fc_size,
                                 weight_stddev=weight_stddev,
                                 use_relu=True)
        layer_fc2_drop = tf.nn.dropout(layer_fc2,keep_prob)

        layer_fc2 = new_fc_layer(input=layer_fc2_drop,
                                 num_inputs=fc_size,
                                 num_outputs=num_of_params,
                                 weight_stddev=weight_stddev,
                                 use_relu=False)

        return layer_fc2


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   weight_stddev,
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size,filter_size,num_input_channels,num_filters]

    # Create new weights aka. filters with the given shape.
    weights = weight_variable(shape=shape,stddev=weight_stddev)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer,weights


#A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected
# layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can
# be used as input to the fully-connected layer.
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer,[-1,num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat,num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 weight_stddev,
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = weight_variable(shape=[num_inputs,num_outputs],stddev=weight_stddev)
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input,weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def new_biases(length):
    return tf.Variable(tf.constant(0.01,shape=[length]))