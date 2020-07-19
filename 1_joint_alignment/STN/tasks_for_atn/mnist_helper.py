#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:51:38 2018

@author: Asher
"""

import numpy as np
import os
import pathlib
import struct
from PIL import Image
import atn_helpers
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#Create an aligned mnist dataset, from the old mnist data.
def create_new_mnist(aligned_data, height, width, path):
    
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) #create the path folder if doesnt exist
    
    files_headers = ["train", "t10k"]
    for ind in range(len(files_headers)):
        data_image = bytearray()
        data_label = bytearray()
        
        images = aligned_data[ind*2]
        lables = aligned_data[ind*2+1]
        for i in range(images.shape[0]):
            im = np.reshape(images[i,:], (height,width))
            img = Image.fromarray(im)
            pixel = img.load()
#           
            for x in range(0,width):
                for y in range(0,height):
                    data_image.append(int(pixel[y,x]))
            data_label.append(lables[i]) # labels start (one unsigned byte each)
            
        data_label,data_image = create_headers(images,data_label,data_image,width)
        
        with open(path+files_headers[ind]+'-images-idx3-ubyte', 'wb') as f:
            for i in data_image:
                f.write(bytes((i,)))
        with open(path+files_headers[ind]+'-labels-idx1-ubyte', 'wb') as f:
            for i in data_label:
                f.write(bytes((i,)))
    

def create_headers(images,data_label,data_image,width):
        
        hexval = "{0:#0{1}x}".format(images.shape[0],6) # number of files in HEX
        picval = "{0:#0{1}x}".format(width,6) # number of files in HEX
         # header for label array
    
        header = bytearray()
        header.extend([0,0,8,1,0,0])
        header.append(int('0x'+hexval[2:][:2],16))
        header.append(int('0x'+hexval[2:][2:],16))
        
        data_label = header + data_label
        
        
        header.append(int('0',16))
        header.append(int('0',16))
        header.append(int('0x'+picval[2:][:2],16))
        header.append(int('0x'+picval[2:][2:],16))
        header.append(int('0',16))
        header.append(int('0',16))
        header.append(int('0x'+picval[2:][:2],16))
        header.append(int('0x'+picval[2:][2:],16))
        
        # additional header for images array
        header[3] = 3 # Changing MSB for image data (0x00000803)
        data_image = header + data_image   
                
        return data_label,data_image
    
    


#Get regular mnist data
def read_normal_mnist(dataset, path):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise (ValueError, "dataset must be 'testing' or 'training'")
        
    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbls = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, n_rows, n_cols = struct.unpack(">IIII", fimg.read(16))
        print("magic={}, num={}, n_rows={}, n_cols={}".format(magic, num, n_rows, n_cols))
        imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), n_rows, n_cols)
    imgs = np.reshape(imgs, (-1, n_rows*n_cols))
    lbls = np.reshape(lbls, (-1, 1))
    return imgs, lbls


def get_digit_data(digit_to_align,batch_size,height,width,path_to_reg_mnist,path_to_new_mnist,test_the_alignment_process, use_cluttered_mnist=False):
    if use_cluttered_mnist:
        mnist_cluttered = np.load(path_to_reg_mnist)
        X_train = mnist_cluttered['X_train']
        y_train = mnist_cluttered['y_train']
        X_train,y_train = update_x_y(X_train,y_train,digit_to_align,batch_size)
        X_test = mnist_cluttered['X_test']
        y_test = mnist_cluttered['y_test']
        X_test,y_test = update_x_y(X_test,y_test,digit_to_align,batch_size)
    else:
        X_train, y_train = read_normal_mnist(dataset = "training", path = path_to_reg_mnist)
        X_train,y_train = update_x_y(X_train,y_train,digit_to_align,batch_size)
        # For testing purposes we'll check the test_the_alignment_process flag.
        # Which creates a training data, where we just took some of the training images and transformed them randomly
        # with affine transformations. We take the test data to be the same as the training data.
        # We want to see that the loss error is close to zero, i.e. we manged to rotated
        # the all the different rotated images back to a fixed position.  
        if test_the_alignment_process:
            X_train = atn_helpers.tranformations_helper.transform_input_images(X_train, height, width, batch_size) 
            y_train = y_train[0:X_train.shape[0]]
            X_test, y_test = (X_train,y_train)
        else:
            X_test, y_test = read_normal_mnist(dataset = "testing", path = path_to_reg_mnist)
            X_test,y_test = update_x_y(X_test,y_test,digit_to_align,batch_size)
        
    return X_train, y_train, X_test, y_test



# In the alignment task we want our network to allign all images of one digit.
# So we will work with only one digit. Need to take only the images of that digit.
def update_x_y(X,y,digit_to_align, batch_size):
    print("original x shape: {}".format(X.shape))
    if digit_to_align is not None: 
        digit_locations = np.where(y[:,0] == digit_to_align)[0]
    else: # we are running mnist classification for all digits in dataset
        digit_locations = np.where(np.logical_and(y>=0, y<=9))[0]
    print("The digit was located {} times".format(digit_locations.shape[0]))
    #We'll find the biggenst number which is >= mages.shape[0]
    #and that divides by batch_size, so that we'll always use the same batch size.
    batch_size_multiplier = np.floor(digit_locations.shape[0]/batch_size)
    num_of_training_data = int(batch_size*batch_size_multiplier)
    digit_locations = digit_locations[0:num_of_training_data]
    X = np.asanyarray([X[i,:] for i in digit_locations])
    print("x final shape: {}".format(X.shape))
    size = (digit_locations.shape[0])
    y = np.reshape(y[digit_locations], (size, 1))
    print("y final shape: {}".format(y.shape))
    return X,y


#This will use a mnist with a much smaller amount of images for each digit.
#if such a dataset doesnt exist, create it.
#this is so that later we can run mnist classification on it, with the alignment cost, and see
#if this allowes fater converging (and a more convex function)
def create_smaller_mnist(minimal_imgs_per_digit, path_to_small_mnist, path_to_reg_mnist,height,width):
    if not os.path.isdir(path_to_small_mnist):
        pathlib.Path(path_to_small_mnist).mkdir(parents=True, exist_ok=True) #create the path folder if doesnt exist
        all_X_train, all_y_train = read_normal_mnist(dataset = "training", path = path_to_reg_mnist)
        all_X_test, all_y_test = read_normal_mnist(dataset = "testing", path = path_to_reg_mnist)
        aligned_data = [[],[],[],[]]
        for digit in range(10):
            updated_training_imgs = []
            updated_test_imgs = []
            X_train,y_train = update_x_y(all_X_train,all_y_train,digit,1)
            X_train,y_train = take_random_few(X_train,y_train,minimal_imgs_per_digit)
            X_test,y_test = update_x_y(all_X_test,all_y_test,digit,1)
            for im in X_train:
                updated_training_imgs.append(np.reshape(im, (height,width)))
            for im in X_test:
                updated_test_imgs.append(np.reshape(im, (height,width)))
            aligned_digit = [updated_training_imgs,y_train,updated_test_imgs,y_test]
            
            for ind in range(len(aligned_digit)):
                #aligned_digit[ind] = turn_to_list(aligned_digit, ind)
                if ind % 2 == 1:
                    label_list = [digit]*(len(aligned_digit[ind]))
                    aligned_digit[ind] = label_list
                #aligned_digit[ind] = np.array(aligned_digit[ind])
                aligned_data[ind] += aligned_digit[ind]
        for ind in range(len(aligned_data)):
            aligned_data[ind] = np.array(aligned_data[ind])
        #shuffle the digits
        shuffle_data(aligned_data, 0, 1)
        create_new_mnist(aligned_data, height, width, path=path_to_small_mnist)
    
def turn_to_list(aligned_digit, ind):
    arr = aligned_digit[ind]
    arr_list = []
    for i in range(arr.shape[0]):
        arr_list.append(arr[i,:])
    return arr_list    

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

def accuracy(logits, labels):
    labels = tf.to_int64(labels)
    predictions = tf.nn.softmax(logits)
    predictions=tf.argmax(predictions,1)
    return tf.metrics.accuracy(labels=labels, predictions=predictions)

def take_random_few(X_train,y_train,minimal_imgs_per_digit):
      inds = np.random.choice(y_train.shape[0],minimal_imgs_per_digit,False)
      return X_train[inds], y_train[inds]
    
def shuffle_data(aligned_data, img_ind, lbl_ind):
    s = np.arange(np.array(aligned_data[lbl_ind]).shape[0])
    np.random.shuffle(s)
    aligned_data[img_ind] = aligned_data[img_ind][s,:]
    aligned_data[lbl_ind] = aligned_data[lbl_ind][s]