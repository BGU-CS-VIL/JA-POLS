#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:16:52 2018

@author: fredman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:12:14 2018

@author: fredman
"""

########################################################################
###
###  This file creates takes in input images, and alignes them, rturning the
###  alined images, and the alignment data loss and regularization loss.
###  The alignment is achieved by using a pixel-location variance loss function.
###  We take the matrix exponent to form affine deffeomorphism transformations.
###
###  Implementations comment -
###  In order to implement the matrix exponential (and its gradient),
###  there was a need to use the batch_size in order to unstack the parametes before taking their exp.
###  For this I needed to use always the same batch size, so I possibly removed some images from
###  the training or test set, so that the number of images mod batch_size will be zero.
###
########################################################################

import os,sys

sys.path.insert(1,os.path.join(sys.path[0],'..'))
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from STN.atn_helpers.spatial_transformer import transformer
from STN.atn_helpers.tranformations_helper import transfromation_parameters_regressor,transformation_regularization_SE, transformation_regularization_VP, transformation_regularization_WEIGHTS, transformation_regularization_SIMPLE
from STN.tasks_for_atn.matrix_exp import expm
import numpy as np

class alignment_transformer_network:

    def __init__(self,x,requested_transforms,regularizations,batch_size,img_sz,num_channels,num_classes,
                 weight_stddev,activation_func,keep_prob, delta, sigma, num_stn):
        self.X = tf.reshape(x,shape=[-1,img_sz[0] * img_sz[1] * num_channels])  #reshaping to 2 dimensions
        self.requested_transforms = requested_transforms
        self.regularizations = regularizations
        self.img_sz = img_sz
        self.num_channels = num_channels
        self.num_classes = tf.cast(num_classes,tf.int32)
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.weight_stddev = weight_stddev
        self.activation_func = activation_func
        self.sigma = sigma  # for Geman-Mecclure robust function
        self.affine_maps = None
        self.sess = None
        self.x_theta = x
        self.theta = None #tf.constant(0.0, shape=[6, self.batch_size])  # tf.constant([0.,0.,0.,0.,0.,0.])
        self.theta_exp = None #tf.constant(0.0, shape=[6,self.batch_size])
        self.transformations_regularizers = tf.constant([0.,0.,0.,0.,0.,0.])
        self.averages_prev = tf.constant(0.0, shape=[img_sz[0], img_sz[1], num_channels-1])
        self.cnt = tf.constant(0.)
        self.num_stn = num_stn

        # Out of 28.8M pixels in the batch:
        #       19.7M are originally nan pixels (diff is less than 1e-3, 9.4 of them are equal to 0)
        #       9M are non-nan pixels
        # How many non-nan pixels are smaller than delta?
        #       delta=0.5: 28.8M - 19.7M = 9M (100% will remain l2)
        #       delta=0.1: 27.5M - 19.7M = 7.8M (86% will remain l2)
        #       delta=0.05: 25.5M - 19.7M = 5.8M (64% will remain l2)
        #       delta=0.025: 23.6M - 19.7M = 3.9M (43% will remain l2)
        #       delta=0.01: 21.2M - 19.7M = 1.5M (16% will remain l2)
        #       delta=0.005: 20.5M - 19.7M = 0.8M (8% will remain l2)
        self.delta = delta

    def stn_diffeo(self):
        with tf.variable_scope("atn"):
            x_tensor = tf.reshape(self.X,[-1,self.img_sz[0],self.img_sz[1],self.num_channels])
            # x_tensor = tf.Print(x_tensor,[x_tensor],message="x_tensor: ",summarize=100)
            c = tf.reduce_mean(tf.boolean_mask(x_tensor, tf.is_finite(x_tensor)), 0)
            # c = tf.Print(c,[c],message="c: ",summarize=100)

            # self.theta, self.affine_maps, d2 = transfromation_parameters_regressor(self.requested_transforms,self.X,
            #                                                                  self.keep_prob,self.img_sz,self.weight_stddev,self.num_channels,self.activation_func)
            #
            # self.theta = tf.Print(self.theta,[self.theta],message="self.theta: ",summarize=100)
            # out_size = (self.img_sz[0], self.img_sz[1])
            # self.theta_exp = expm(-self.theta)  # compute matrix exponential on {-theta}
            # # self.theta_exp = tf.Print(self.theta_exp,[self.theta_exp],message="theta_exp: ", summarize=100)
            # x_theta, d = transformer(x_tensor, self.theta_exp, out_size)
            # #to avoid the sparse indexing warning, comment the next line, and uncomment the one after it.
            # self.x_theta = tf.reshape(x_theta,shape=[-1,self.img_sz[0],self.img_sz[1],self.num_channels])
            # d.update({'params':d2['params']})

            # Working with recurrent STN: get self.theta and self.theta_exp in shape: [num_STN, batch_sz, 6]
            d = c
            self.x_theta, self.theta, self.theta_exp = transfromation_parameters_regressor(self.requested_transforms, self.X,
                                                                                 self.keep_prob,self.img_sz,
                                                                                 self.weight_stddev,self.num_channels,
                                                                                 self.activation_func, self.num_stn)

            return self.x_theta, d, c

    def compute_alignment_loss(self,lables_one_hot=None):
        with tf.variable_scope("atn"):
            self.lables_one_hot = lables_one_hot
            self.alignment_loss, a, b = self.alignment_loss()
            return self.alignment_loss, a, b

    def alignment_loss(self):
        # ------------------------ Our loss (with W) ------------------------------------------------------
        img = tf.slice(self.x_theta, [0, 0, 0, 0], [-1, -1, -1, self.num_channels - 1])  # (64, 128, 128, 3)
        mask = tf.slice(self.x_theta, [0, 0, 0, self.num_channels - 1], [-1, -1, -1, -1])  # (64, 128, 128, 1)

        #img = tf.layers.batch_normalization(x_theta_new0)

        #img = tf.multiply(img_slice, mask)  # (64, 128, 128, 3)

        sum_weighted_imgs = tf.reduce_sum(img, 0)
        # sum_weighted_imgs = tf.Print(sum_weighted_imgs, [sum_weighted_imgs], message="sum_weighted_imgs: ", summarize=100)

        sum_weights = tf.reduce_sum(mask, 0)
        sum_weights = tf.concat([sum_weights,sum_weights,sum_weights], 2)
        # sum_weights = tf.Print(sum_weights, [sum_weights], message="sum_weights: ", summarize=100)

        # averages = tf.where(tf.less(sum_weights, 1e-3), tf.zeros_like(sum_weighted_imgs), tf.divide(sum_weighted_imgs, sum_weights+1e-7))  #sum_weights+1e-7
        # averages = tf.Print(averages, [averages], message="averages: ", summarize=100)

        # "Recursive" average:
        average_batch = tf.where(tf.less(sum_weights, 1e-3), tf.zeros_like(sum_weighted_imgs), tf.divide(sum_weighted_imgs, sum_weights+1e-7))  #sum_weights+1e-7
        averages_curr = tf.divide(tf.multiply(self.cnt, self.averages_prev) + average_batch, tf.add(self.cnt, 1))
        # self.averages_prev = averages_curr
        # self.cnt = tf.add(self.cnt, 1)

        weighted_diff = tf.multiply(mask, tf.subtract(img, averages_curr))

        # square_weighted_diff = tf.square(weighted_diff)
        huber_diff = tf.where(tf.less(tf.abs(weighted_diff), self.delta), 0.5*tf.square(weighted_diff), self.delta*tf.abs(weighted_diff)-0.5*(self.delta**2))

        huber_diff_robust = huber_diff / (huber_diff + self.sigma ** 2)

        loss_sum_per_pixel = tf.reduce_sum(huber_diff_robust, 0)
        alignment_loss = tf.reduce_sum(loss_sum_per_pixel)
        alignment_loss =  alignment_loss/tf.reduce_sum(mask)   #  alignment_loss / (self.img_sz[0] * self.img_sz[1] * self.num_channels)
        # alignment_loss = tf.reduce_sum(alignment_loss / (alignment_loss + self.sigma ** 2))
        # alignment_loss = alignment_loss / (alignment_loss + self.sigma ** 2)

        a = tf.reduce_mean(tf.boolean_mask(self.x_theta, tf.is_finite(self.x_theta)), 0)
        # a = tf.Print(a,[a],message="a: ",summarize=100)
        b = tf.reduce_sum(mask) # need to remove
        # b = tf.Print(b,[b],message="b: ",summarize=100)

        return alignment_loss, a, b

    def alignment_loss_median(self):
        # ------------------------ Our loss (with W) ------------------------------------------------------
            # Compute median per pixel-stack:
        num_of_img_pixels = self.img_sz[0] * self.img_sz[1] * (self.img_sz[2]-1)
        img = tf.slice(self.x_theta, [0, 0, 0, 0], [-1, -1, -1, self.num_channels - 1])  # Shape: (bs, h, w, 3)
        mask = tf.slice(self.x_theta, [0, 0, 0, self.num_channels - 1], [-1, -1, -1, -1])  # Shape: (bs, h, w, 1)
        bool_mask = tf.logical_not(tf.equal(mask, 0))
        bool_mask = tf.concat([bool_mask, bool_mask, bool_mask], 3)
        neg_val = tf.ones_like(img) * -200.
        x_theta_tmp = tf.where(bool_mask, img, neg_val)  # Shape: (bs_sz, h, w, 3)
        # a = tf.reduce_max(x_theta_tmp)
        # a = tf.Print(a,[a],message="a: ",summarize=100)

        batch_elements = tf.transpose(tf.reshape(x_theta_tmp, [-1, num_of_img_pixels]))  # Shape: (h*w*3, bs)
        batch_elements = tf.concat([batch_elements, tf.zeros([num_of_img_pixels, 1])], 1)

        medians_pixels_stack = tf.map_fn(
            lambda x: tf.contrib.distributions.percentile(tf.boolean_mask(x, x > -10.), q=50., axis=[0]), batch_elements, dtype=tf.float32)  # Shape: (h*w*3, )
        # -- Instead, perform median on all values including irrelevant pixels:
        # medians_pixels_stack = tf.contrib.distributions.percentile(batch_elements, q=50., axis=[1]) # Shape: (h*w*3, )
        medians_in_img_shape = tf.reshape(medians_pixels_stack, [self.img_sz[0], self.img_sz[1], (self.img_sz[2]-1)])  # Shape: (h, w, 3)

            # Compute loss per pixel-stack:
        weighted_diff = tf.multiply(mask, tf.subtract(img, medians_in_img_shape))   # Shape: (bs, h, w, 3)
        # square_weighted_diff = tf.square(weighted_diff)
        huber_diff = tf.where(tf.less(tf.abs(weighted_diff), self.delta), 0.5*tf.square(weighted_diff), self.delta*tf.abs(weighted_diff)-0.5*(self.delta**2))
        huber_diff_robust = huber_diff #/ (huber_diff + self.sigma ** 2)

            # Compute total loss:
        loss_sum_pixel_stack = tf.reduce_sum(huber_diff_robust, 0)
        alignment_loss = tf.reduce_sum(loss_sum_pixel_stack)
        alignment_loss = alignment_loss/tf.reduce_sum(mask)   #  alignment_loss / (self.img_sz[0] * self.img_sz[1] * self.num_channels)
        # alignment_loss = tf.reduce_sum(alignment_loss / (alignment_loss + self.sigma ** 2))
        # alignment_loss = alignment_loss / (alignment_loss + self.sigma ** 2)

        a = tf.reduce_mean(tf.boolean_mask(self.x_theta, tf.is_finite(self.x_theta)), 0)
        # # a = tf.Print(a,[a],message="a: ",summarize=100)
        b = tf.reduce_sum(mask) # need to remove
        # b = tf.Print(b,[b],message="b: ",summarize=100)

        return alignment_loss, a, b

    def compute_transformations_regularization(self, regulator_type):

        theta = tf.reduce_sum(self.theta, 0)

        if regulator_type == 'SE':
            self.transformations_regularizers = transformation_regularization_SE(theta)
        elif regulator_type == 'WEIGHTS':
            self.transformations_regularizers = transformation_regularization_WEIGHTS(self.affine_maps, self.regularizations)  #give a diffrent penalty to each type of transformation magnituted
        elif regulator_type == 'VP':
            self.transformations_regularizers = transformation_regularization_VP(theta)
        elif regulator_type == 'SIMPLE':
            self.transformations_regularizers = transformation_regularization_SIMPLE(theta)
        return self.transformations_regularizers


    def get_theta_exp(self):
        return self.theta_exp  #(tf.slice(self.theta_exp, [0,0], [-1,6]))