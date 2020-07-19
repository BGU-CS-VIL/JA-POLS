########################################################################
###
###  This file creates a mnist-like dataset for a specific digit,
###  after rotating each of the specific digit's images, to 
###  be alined to a fixed position.
###  This is achieved by using a pixel-location variance loss function.
###  Here we are restricting the transformations to only by rotations.
###  We also take the matrix exponent to form affine deffeomorphism 
###  transformations.
###
###  Implementations comment - 
###  In order to implement the matrix exponential (and its gradient),
###  there was a need to use the batch_size in order to unstack the parametes before taking their exp.
###  For this I needed to use always the same batch size, so I possibly removed some images from
###  the training or test set, so that the number of images mod batch_size will be zero.
###
########################################################################

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_helper
from atn_helpers.tranformations_helper import register_gradient
from ATN import alignment_transformer_network
#from skimage.transform import warp, AffineTransform
from data_provider2 import DataProvider

    

# %% Load data
#height, width = [28,28] #for regular mnist
height, width = [40,40] #for cluttered mnist
def main():
    
    # Here you can play with some parameters.
    n_epochs = 4
    batch_size = 16
    
    num_channels = 3
    # possible trasromations = "r","sc","sh","t","ap","us","fa"
    # see explanations in transformations_helper.py
    requested_transforms = ["t"] # ["t","sc","r","sh"]
    regularizations = {"r":0,"t":0.,"sc":0.5,"sh":0.}
#    requested_transforms = ["fa"]
#    regularizations = {"fa":5.}
    #requested_transforms = ["ane"] #uncomment if you only want a regular stn with no matrix exponential
    
    #for alignemnt alone, relu gives better performance when used in the stn, 
    #but for classifcation, tanh gives better results, so you might want to also try tanh. 
    activation_func = "relu"
    
    # param my_learning_rate
    # Gets good results with 1e-4. You can also set the weigts in the transformations_helper file 
    # (good results also with 1e-4 initialization)
    my_learning_rate = 1e-3
    weight_stddev = 5e-3
    
    #param use_small_mnist
    #if is true, we will use a mnist with a much smaller amount of images for each digit.
    #if such a dataset doesnt exist, create it.
    #this is so that later we can run mnist classification on it, with the alignment cost, and see
    #if this allowes fater converging (and a more convex function)
    use_small_mnist = False
    minimal_imgs_per_digit = 10
    
    use_cluttered_mnist = True
    
    # if we are adding the alignment loss, which is very big, to the mnist loss, 
    # we will multiply the mnist loss by the mnist_regularizer.
    mnist_regularizer = 50 
    
    # param test_the_alignment_process
    # If true will transform a few images in the training data many times
    # and we'll test if the alignment works. 
    test_the_alignment_process = False 
    
    prepare_figure()
    
    #measure the time
    start_time = time.time() 
    
    mnist_path = path_to_reg_mnist #will tell us which dataset to take the digit from later
    if use_small_mnist:
        print("will use a smaller verion of mnist with {} images".format(minimal_imgs_per_digit))
        mnist_path = path_to_small_mnist
        mnist_helper.create_smaller_mnist(minimal_imgs_per_digit, path_to_small_mnist+"/", path_to_reg_mnist, height,width)
        
    if use_cluttered_mnist:
        mnist_path = path_to_cluttered_mnist
    
    # Load data and take only the desired digit images
    params = (None, batch_size, height,width, mnist_path, path_to_new_mnist, test_the_alignment_process, use_cluttered_mnist)

    # !!! MINE !!!
    img_sz = 128
    video_name = "movies/BG.mp4"
    data = DataProvider(video_name,img_sz)
    X_train = data.next_batch(batch_size, 'train')
    y_train = data.next_batch(batch_size,'train')
    X_test = data.next_batch(batch_size,'train')
    y_test = data.next_batch(batch_size,'train')
    #X_train,y_train,X_test,y_test = mnist_helper.get_digit_data(*params)

    loss,mnist_loss,mnist_accuracy,alignment_loss,transformations_regularizer,logits,logits2,transformations,b_s,x,y,keep_prob,optimizer = computational_graph(my_learning_rate,requested_transforms,batch_size,activation_func,regularizations,weight_stddev,mnist_regularizer,num_channels)
        
    # We now create a new session to actually perform the initialization the variables:
    params = (X_train,y_train,y_test,X_test,n_epochs,batch_size,loss,mnist_loss,mnist_accuracy,alignment_loss,transformations_regularizer,logits,logits2,transformations,b_s,x,y,keep_prob,optimizer,start_time)
    run_session(*params)
    
    #measure the time
    # Set the precision.
    duration = time.time() - start_time
    print("Total runtime is "+ "%02d" % (duration) + " seconds.")



# %%
    
def computational_graph(my_learning_rate,requested_transforms,batch_size,activation_func,regularizations,weight_stddev,mnist_regularizer=1,num_channels=1):
    x = tf.placeholder(tf.float32, [None, height*width])# input data placeholder for the atn layer
    y = tf.placeholder(tf.float32, [None, 1])
    #batch size
    b_s = tf.placeholder(tf.float32, [1,])
    
    # Since x is currently [batch, height*width], we need to reshape to a
    # 4-D tensor to use it in a convolutional graph.  If one component of
    # `shape` is the special value -1, the size of that dimension is
    # computed so that the total size remains constant.  Since we haven't
    # defined the batch dimension's shape yet, we use -1 to denote this
    # dimension should not change size.
    keep_prob = tf.placeholder(tf.float32)
    atn = alignment_transformer_network(x, requested_transforms, regularizations, batch_size, width, num_channels, weight_stddev, activation_func, keep_prob)
    logits, transformations, alignment_loss, transformations_regularizer = atn.atn_layer()
    
    logits2 = cnn_graph(logits, my_learning_rate, keep_prob)
    
    loss,mnist_loss,mnist_accuracy = compute_final_loss(logits2, y, alignment_loss, transformations_regularizer,mnist_regularizer)
#    loss = mnist_loss + transformations_regularizer

    opt = tf.train.AdamOptimizer(learning_rate=my_learning_rate)
    optimizer = opt.minimize(loss)
    #grads = opt.compute_gradients(loss, [b_fc_loc2])
    
    return loss,mnist_loss/mnist_regularizer,mnist_accuracy,alignment_loss,transformations_regularizer,logits,logits2,transformations,b_s,x,y,keep_prob,optimizer
    
def compute_final_loss(logits2, y, alignment_loss, transformations_regularizer, mnist_regularizer=1):
    mnist_loss = mnist_helper.loss(logits2, y) * mnist_regularizer
    mnist_accuracy = mnist_helper.accuracy(logits2, y)
    loss = mnist_loss + alignment_loss + transformations_regularizer
    return loss,mnist_loss,mnist_accuracy

# %%
    
def cnn_graph(input_images, my_learning_rate,keep_prob):

  # Convolutional Layer 1.
  filter_size1 = 5  
  num_filters1 = 16  

  # Convolutional Layer 2.
  filter_size2 = 5   
  num_filters2 = 36     
  
  # Fully-connected layer.
  fc_size = 128  

  # Number of colour channels for the images: 1 channel for gray-scale.
  num_channels = 1

  # Number of classes, one class for each of 10 digits.
  num_classes = 10

  # The convolutional layers expect x to be encoded as a 4-dim tensor so we have to 
  #reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
  input_images = tf.reshape(input_images, shape=[-1,height,width,1])

  layer_conv1, weights_conv1 = \
    new_conv_layer(input=input_images,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

  layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

  #before sending the input to a non-convolutional layer, need to re-flatten it
  #(need to undo what we did in the reshape before the convolutional layer above)
  layer_flat, num_features = flatten_layer(layer_conv2)

  layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
  layer_fc1_drop = tf.nn.dropout(layer_fc1, keep_prob)

  layer_fc2 = new_fc_layer(input=layer_fc1_drop,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

  logits = layer_fc2
  return logits
    



def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
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
    return layer, weights

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
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

    

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# %%

def run_session(data, X_train,y_train,y_test,X_test,n_epochs,batch_size,loss,mnist_loss,mnist_accuracy,alignment_loss,transformations_regularizer,logits,logits2,transformations,b_s,x,y,keep_prob,optimizer,start_time):
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    #find the indexes needed for splitting the train and test sets into batchs with the desired batch size
    iter_per_epoch,indices,test_iter_per_epoch,test_indices = prepare_splitting_data(X_train,X_test,batch_size)
    
    for epoch_i in range(n_epochs):
        for iter_i in range(iter_per_epoch):
            batch_xs = data.next_batch(batch_size, 'train') # X_train[indices[iter_i]:indices[iter_i+1]]
            batch_ys = data.next_batch(batch_size, 'train') #y_train[indices[iter_i]:indices[iter_i+1]]
            batch_size = batch_ys.size
            
            
            loss_val,theta_val = sess.run([loss,transformations],
                            feed_dict={
                                b_s: [batch_size],
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            if iter_i % 200 == 0:
                print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss_val))
                print ("theta row 1 is: "+str(theta_val[0,:]))
    
            sess.run(optimizer, feed_dict={
                b_s: [batch_size], x: batch_xs, y: batch_ys, keep_prob: 0.8})
        
        
        #Find accuracy on test data
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nrunning test data...")
        accuracy = 0.
        all_loss = 0.
        mnist_loss_avg = 0.
        for iter_i in range(test_iter_per_epoch):
            batch_xs = X_test[test_indices[iter_i]:test_indices[iter_i+1]]
            batch_ys = y_test[test_indices[iter_i]:test_indices[iter_i+1]]
            batch_size = batch_ys.size
            
            
            loss_val,mnist_loss_val,mnist_accuracy_val,alignment_loss_val,transformations_regularizer_val = sess.run([loss,mnist_loss,mnist_accuracy,alignment_loss,transformations_regularizer],
                                    feed_dict={
                                                 b_s: [batch_size],
                                                 x: batch_xs,
                                                 y: batch_ys,
                                                 keep_prob: 1.0
                                               })
            
            all_loss += loss_val
            mnist_loss_avg += mnist_loss_val
            accuracy += mnist_accuracy_val[0]
        all_loss /= test_iter_per_epoch
        mnist_loss_avg /= test_iter_per_epoch
        accuracy /= test_iter_per_epoch
        #print ("theta row 1 is: "+str(theta_val[0,:]))
        #print ("theta row 10 is: "+str(theta_val[9,:]))
        print('Combined Test Loss (%d): ' % (epoch_i+1) + str(all_loss))
        print('alignment loss {}   ||||  transformations regularization {}: '.format(alignment_loss_val,transformations_regularizer_val[0]))
        print('Classification Test Loss is (%d): ' % (epoch_i+1) + str(mnist_loss_avg))
        print('Classification Test Accuracy is (%d): ' % (epoch_i+1) + str(accuracy))
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        if  np.isnan(all_loss):
            duration = time.time() - start_time
            print("Total runtime is "+ "%02d" % (duration) + " seconds.")
            raise SystemExit
        
    #show some of the test data before and after running the model which was learned
    all_test_imgs = None#Find accuracy on test data
    print("\n\nPreparing test images...")
    for iter_i in range(test_iter_per_epoch):
        batch_xs = X_test[test_indices[iter_i]:test_indices[iter_i+1]]
        batch_ys = y_test[test_indices[iter_i]:test_indices[iter_i+1]]
        batch_size = batch_ys.size
        
        loss_val,test_imgs = sess.run([loss,logits],
                                feed_dict={
                                             b_s: [batch_size],
                                             x: batch_xs,
                                             y: batch_ys,
                                             keep_prob: 1.0
                                           })        
        if all_test_imgs is not None:
            all_test_imgs = np.concatenate((all_test_imgs,test_imgs))
        else:
            all_test_imgs = test_imgs
    show_test_imgs(all_test_imgs,X_test,height,width)
    
    sess.close()
    

# %%

#def layer_grid_summary(name, var, image_dims, BATCH_SIZE):
#    prod = np.prod(image_dims)
#    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod], [BATCH_SIZE*28, BATCH_SIZE*28], image_dims, 1))
#    return tf.summary.image(name, grid)

def create_summaries(loss, x, output, BATCH_SIZE):
    writer = tf.summary.FileWriter("../logs")
    tf.summary.scalar("Loss", loss)
#    layer_grid_summary("Input", x, [28, 28], BATCH_SIZE)
#    layer_grid_summary("Output", output, [28, 28], BATCH_SIZE)
    return writer, tf.summary.merge_all()

def prepare_splitting_data(X_train,X_test,batch_size):
    train_size = X_train.shape[0]
    iter_per_epoch = int(train_size/batch_size)
    indices = np.linspace(0, train_size, iter_per_epoch+1)
    indices = indices.astype('int')
    
    test_size = X_test.shape[0]
    test_iter_per_epoch = int(test_size/batch_size)
    test_indices = np.linspace(0, test_size, test_iter_per_epoch+1)
    test_indices = test_indices.astype('int')
    
    return iter_per_epoch,indices,test_iter_per_epoch,test_indices


def prepare_figure():
#    plt.figure(17, figsize=(figure_size,figure_size))
#    plt.clf()
    plt.subplots_adjust(left=0.14, bottom=-1., right=0.8 , top=1.)
    #plt.rcParams['figure.figsize'] = (200., 150.)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'jet'
    


#show the first 10 figures of the test data after running the model which was learned
def show_test_imgs(all_test_imgs,X_test,height,width,show_num_imgs=40):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("The newly created image set is of shape {}".format(all_test_imgs.shape))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    inds = np.random.choice(X_test.shape[0],show_num_imgs,False)
    X_test = X_test[inds]
    all_test_imgs = all_test_imgs[inds]
    PlotImages(np.reshape(X_test,(-1,height,width)), np.reshape(all_test_imgs,(-1,height,width)), show_num_imgs)
        
    
def PlotImages(before_imgs, after_imgs, num_imgs):
    plt_idx = 1
    for ind in range(num_imgs):
        img = before_imgs[ind]
        plt_idx = show_im(img, num_imgs, plt_idx, '')
        img = after_imgs[ind]
        plt_idx = show_im(img, num_imgs, plt_idx, '')
    plt.show()
    

def show_im(img, num_imgs, plt_idx, title):
    plt.subplot(num_imgs/2, 8, plt_idx)
    plt.imshow(img)
    #plt.title(title, loc='left')
    return plt_idx+1
    
#Run one forward pass again on the training data, in ordr to create a transformed mnist data
def prepare_new_mnist(sess,X_train,y_train,iter_per_epoch,indices,batch_ys,batch_xs,loss,logits,transformations,b_s,x,y,keep_prob):
    all_training_imgs = None
    for iter_i in range(iter_per_epoch):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = y_train[indices[iter_i]:indices[iter_i+1]]
        batch_size = batch_ys.size
        
        loss_val,training_imgs,theta_val = sess.run([loss,logits,transformations],
                                    feed_dict={
                                                 b_s: [batch_size],
                                                 x: batch_xs,
                                                 y: batch_ys,
                                                 keep_prob: 1.0
                                               })
        
        if all_training_imgs is not None:
            all_training_imgs = np.concatenate((all_training_imgs,training_imgs))
        else:
            all_training_imgs = training_imgs
    return all_training_imgs
    

# %%
        
if __name__ == '__main__':
    #register the gradient for matrix exponential
    register_gradient()
    
    #some global params which will probably not be changed
    path_to_reg_mnist = "../../data/mnist/regular_data"
    path_to_small_mnist = "../../data/mnist/minimal_data"
    path_to_new_mnist = "../../data/mnist/affine_deffeomorphism/"
    path_to_cluttered_mnist = "../../data/mnist/cluttered_mnist/mnist_sequence1_sample_5distortions5x5.npz"
    update_transformed_mnist = False
    rows = 10
    cols = 2
    figure_size = 5*rows
    
    main()
