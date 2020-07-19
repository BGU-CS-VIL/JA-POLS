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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.insert(1, os.path.join(sys.path[0], '..'))
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
from STN.ATN import alignment_transformer_network
from STN.atn_helpers.tranformations_helper import register_gradient
from STN.tasks_for_atn.data_provider_STN import DataProvider
from utils.Plots import open_figure, PlotImages
import cv2
from utils.image_warping import warp_image
from numpy.linalg import inv
import config

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# --------- SETTINGS --------------

mypath = config.paths['my_path']
input_data_path = 'data/'
model_path = 'STN/model/model.ckpt'

mydevice = config.stn['device']
load_model = config.stn['load_model']
num_stn = config.stn['num_stn']
iter_per_epoch = config.stn['iter_per_epoch']
batch_size = config.stn['batch_size']
weight_stddev = config.stn['weight_stddev']
activation_func = config.stn['activation_func']
delta = config.stn['delta']
sigma = config.stn['sigma']
align_w = config.stn['align_w']
regul_w = config.stn['regul_w']
regul_w2 = config.stn['regul_w2']
alignment_reg = config.stn['alignment_reg']
regulator_reg = config.stn['regulator_reg']
first_regulator = config.stn['first_regulator']
second_regulator = config.stn['second_regulator']
lrn_rate = config.stn['lrn_rate']

# --------- END SETTINGS -----------

loss_list = []
loss_align_list = []
loss_regul_list = []

# %% Load data
def main_STN():
    #
    print("\nStart STN run...")

    print("\nNumber of iterations: ", iter_per_epoch)
    print("\nBatch size (for training): ", batch_size)

    data = DataProvider(mypath + input_data_path)

    # Here you can play with some parameters.
    n_epochs = 1
    num_channels = data.img_sz[2]

    # possible trasromations = "r","sc","sh","t","ap","us","fa"
    # see explanations in transformations_helper.py
    requested_transforms = ["t"] #["r","t","sc","sh"]
    regularizations = {"r":100.0, "t":100.0, "sc":10000.0, "sh":1000.0, "fa":100.0}  # 0.000005

    register_gradient()

    # param my_learning_rate
    # Gets good results with 1e-4. You can also set the weigts in the transformations_helper file
    # (good results also with 1e-4 initialization)
    # my_learning_rate = 1e-5

    #measure the time
    start_time = time.time()

    device = mydevice
    with tf.device(device):  #greate the graph
        loss,loss2,x_theta,theta_exp,b_s,x,keep_prob,optimizer, optimizer2,alignment_loss, trans_regularizer, trans_regularizer2, atn, a, b, d, c, learning_rate = computational_graph(
                                                                                      requested_transforms,
                                                                                      regularizations,activation_func,
                                                                                      weight_stddev,num_channels,data.img_sz, num_stn)

    # We now create a new session to actually perform the initialization the variables:
    params = (data,iter_per_epoch,n_epochs,batch_size,loss,loss2,x_theta,theta_exp,b_s,x,keep_prob,optimizer,optimizer2,
              start_time,data.img_sz,num_channels, alignment_loss, trans_regularizer, trans_regularizer2, a, b, d, c, atn, learning_rate, mypath)
    run_session(*params)

    duration = time.time() - start_time
    print("Total runtime is " + "%02d" % (duration) + " seconds.")


def computational_graph(requested_transforms,regularizations,activation_func,weight_stddev,
                        num_channels,img_sz, num_stn):

    x = tf.placeholder(tf.float32, [None, img_sz[0] * img_sz[1] * num_channels])  # input data placeholder for the atn layer

    #batch size
    b_s = tf.placeholder(tf.float32, [1, ])
    keep_prob = tf.placeholder(tf.float32)

    # ------------- Learn theta: ----------------
    atn = alignment_transformer_network(x, requested_transforms, regularizations, b_s, img_sz, num_channels, 1, weight_stddev, activation_func, 1, delta, sigma, num_stn)
    x_theta, d, c = atn.stn_diffeo()
    theta_exp = atn.get_theta_exp()

    # ------------- Compute loss: ----------------
    trans_regularizer = atn.compute_transformations_regularization(first_regulator)
    trans_regularizer2 = atn.compute_transformations_regularization(second_regulator)
    alignment_loss, a, b = atn.compute_alignment_loss()
    alignment_loss = alignment_loss * alignment_reg
    trans_regularizer =  trans_regularizer * regulator_reg
    trans_regularizer2 = trans_regularizer2 * regulator_reg
    loss = compute_final_loss(alignment_loss, trans_regularizer, trans_regularizer2, num_channels, img_sz)
    loss2 = compute_final_loss(alignment_loss, trans_regularizer, trans_regularizer2,num_channels,img_sz)

    # diff = tf.subtract(theta_gt,theta)
    # sqaure_sum = tf.reduce_sum(tf.square(diff),0)  # reduce_mean
    # # loss = tf.reduce_sum(tf.abs(tf.subtract(theta_gt,theta)))
    # loss = tf.reduce_sum(sqaure_sum / (sqaure_sum + sigma ** 2))

    # ------------- Prepare Optimizer: ----------------
    # Adaptive learning rate
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,100000,0.96,staircase=True)
    # optimizer = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step))
    # optimizer2 = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2,global_step=global_step))

    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)  # tf.train.RMSPropOptimizer, tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer, tf.train.MomentumOptimizer(lrn, 0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(loss)
        optimizer2 = opt.minimize(loss2)

    return loss,loss2,x_theta,theta_exp,b_s,x,keep_prob,optimizer, optimizer2,alignment_loss, trans_regularizer, trans_regularizer2, atn, a, b, d, c, learning_rate


def compute_final_loss(alignment_loss,regularizer1, regularizer2, num_channels,img_sz):
    #alignment_loss = alignment_loss / (img_sz[0] * img_sz[1] * num_channels)
    return (align_w * alignment_loss) + (regul_w * regularizer1) + (regul_w2 * regularizer2)


def run_session(data,iter_per_epoch,n_epochs,batch_size,loss,loss2,x_theta,theta_exp,b_s,x,keep_prob,
                optimizer,optimizer2,start_time, img_sz,num_channels, alignment_loss, trans_regularizer, trans_regularizer2, a, b, d, c, atn, learning_rate, mypath):
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if load_model:
        saver.restore(sess, model_path)
        print('Model restored.')

    # --------------------------- Train step: --------------------------------

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nstart training...")
    for iter_i in range(iter_per_epoch):
        batch_x = data.next_batch(batch_size, 'train')
        # params = d['params']
        # params = tf.Print(params,[params],message="params: ",summarize=100)
        loss_val, theta_exp_val, alignment_loss_val, trans_regularizer_val, trans_regularizer_val2, a_val, b_val, c_val = sess.run([loss, theta_exp, alignment_loss, trans_regularizer, trans_regularizer2, a,b, c],
                                      feed_dict={
                                          b_s:[batch_size],
                                          x:batch_x,
                                          keep_prob:1.0
                                      })
        loss_list.append(loss_val)
        loss_align_list.append(alignment_loss_val)
        loss_regul_list.append(trans_regularizer_val)
        if iter_i % 20 == 0:
            print('Iteration: ', iter_i, ' Loss: ', loss_val, ' Alignment Loss: ', alignment_loss_val, ' Regulation Loss: ', trans_regularizer_val, ' Regulation2 Loss: ', trans_regularizer_val2)
            print("theta row 1 is: " + str(theta_exp_val[0,0,:]))
            saver.save(sess,model_path)

        sess.run(optimizer,feed_dict={b_s:[batch_size],x:batch_x,keep_prob:1.0,learning_rate:lrn_rate})

    # --------------------------- Test step: ------------------------------------

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nprepare residual transformations...")
    batch_x = data.next_batch(-1, 'test')
    x_theta_test = np.zeros((1, img_sz[0], img_sz[1], img_sz[2]))
    theta_exp_test = np.zeros((1, 6))
    bs = 10
    for j in range(0, len(batch_x), bs):
        if j+bs < len(batch_x):
            current_batch = batch_x[j:j+bs, ...]
        else:
            current_batch = batch_x[j:, ...]

        new_sz = len(current_batch)
        loss_val, x_theta_val, theta_exp_val, a_val2 = sess.run([loss, x_theta, theta_exp, a],
                                                feed_dict={
                                                    b_s:[new_sz],
                                                    x:current_batch,
                                                    keep_prob:1.0
                                                })

        # compose the num_STN transformations into one:
        theta_exp_val = compose_transformations(theta_exp_val, new_sz)
        theta_exp_test = np.concatenate((theta_exp_test, theta_exp_val), axis=0)

        x_theta_test = np.concatenate((x_theta_test, x_theta_val), axis=0)

    # Save residual transformations in a file:
    x_theta_test = x_theta_test[1:, ...]
    theta_exp_test = theta_exp_test[1:, ...]
    np.save(mypath + input_data_path + "AFFINE_residual.npy", theta_exp_test)  # shape of theta_exp_test: N x 6
    print('AFFINE_residual shape: ', theta_exp_test.shape)

    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nNetwork run is done.")

    # Plot the test results
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nPlot results...")
    plot_results(batch_x,x_theta_test,theta_exp_test,img_sz,num_channels,batch_size,loss_list,loss_align_list,loss_regul_list,data.batch_train, mypath)

    sess.close()

    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nProcess is done.")


# input: theta_exp_all, shape: [num_STN, batch_sz, 6]
# output: theta_exp, shape: [batch_sz, 6]
def compose_transformations(theta_exp_all, batch_sz):
    theta_results = np.zeros((batch_sz, 6))
    for i in range(batch_sz):
        trans = theta_exp_all[:, i, :]  # shape: [num_STN, 6]
        T_final = np.eye(3)
        bottom = np.expand_dims(np.array([0,0,1]), axis=0)
        for j in range(trans.shape[0]):
            T = np.concatenate((np.reshape(trans[j], (2, 3)), bottom), axis=0)
            T_final = T_final @ T

        theta_results[i] = np.ravel(T_final[0:2, :])

    return theta_results


def plot_results(batch_x, x_theta_test, theta_exp_test, img_sz, num_channels, batch_size, loss_list, loss_align_list, loss_regul_list, data_orig, mypath):
    plt.clf()
    plt.plot(loss_list)
    plt.plot(loss_align_list)
    plt.plot(loss_regul_list)
    plt.legend(['Total loss', 'Alignment loss','Regulator loss'], loc='best')
    plt.savefig(mypath + '1_joint_alignment/STN/stn_alignment_results/loss.png')

    imgs_trans_all = []
    imgs_trans_all2 = []
    imgs_notrans_all = []
    for i in range(len(batch_x)):
        I_orig = np.reshape(data_orig[i, ...], (img_sz[0], img_sz[1], num_channels))
        I_orig = prepare_nan_img(np.abs(I_orig))
        imgs_notrans_all.append(I_orig)

        I = np.reshape(batch_x[i,...], (img_sz[0], img_sz[1], num_channels))
        I = prepare_nan_img(np.abs(I))
        T = theta_exp_test[i, ...]
        T = np.reshape(T, (2, 3))

        I_t, d = warp_image(I, T, cv2.INTER_CUBIC)
        I_t = np.abs(I_t/np.nanmax(I_t))
        imgs_trans_all.append(I_t)

        # take transform image from the special transformer
        I_t2 = np.reshape(x_theta_test[i, ...], (img_sz[0], img_sz[1], num_channels))
        I_t2 = prepare_nan_img(np.abs(I_t2))
        I_t2 = np.abs(I_t2 / np.nanmax(I_t2))
        imgs_trans_all2.append(I_t2)

    # --------- build panoramic image of original images:------------
    panoramic_img_notrans = np.nanmedian(imgs_notrans_all, axis=0)  # nanmean
    fig3 = open_figure(3, 'Panoramic Image', (3, 2))
    PlotImages(3, 1, 1, 1, [panoramic_img_notrans], [''], 'gray', axis=False, colorbar=False)

    # --------- build panoramic image of transformed images using my warping:--------
    panoramic_img = np.nanmedian(imgs_trans_all, axis=0)  # nanmean
    fig4 = open_figure(4, 'Panoramic Image', (3, 2))
    PlotImages(4, 1, 1, 1, [panoramic_img], [''], 'gray', axis=False, colorbar=False)

    # --------- build panoramic image of transformed images using spacial transformer:----------
    #panoramic_img2 = np.nanmedian(imgs_trans_all2, axis=0)
    #fig5 = open_figure(5,'Panoramic Image',(3,2))
    #PlotImages(5,1,1,1,[panoramic_img2],[''],'gray',axis=False,colorbar=False)

    plt.show()
    fig3.savefig(mypath + '1_joint_alignment/STN/stn_alignment_results/STN_Panorama_initial.png', dpi=1000)
    fig4.savefig(mypath + '1_joint_alignment/STN/stn_alignment_results/STN_Panorama_transformed.png', dpi=1000)
    #fig5.savefig(mypath + '1_joint_alignment/STN/stn_alignment_results/STN_Panorama_Transformed2.png', dpi=1000)


def nan_if(lst):
    new_lst = []
    for i in range(len(lst)):
        I = lst[i]
        I = np.nan_to_num(I)
        I_new = np.where(I <= 1e-04, np.nan, I)
        new_lst.append(I_new)
    return new_lst

# Get image with binary mask (depth=4)
# Returns the image (depth=3) with nans where the mask is 0, and the image values where the mask is 1.s
def prepare_nan_img(img):
    I = img[:, :, 0:3]
    W = img[:, :, 3:]
    W_N = np.concatenate((W, W, W), axis=2)
    I[np.where(W_N < 1)] = np.nan
    return I


def embed_image(img, img_emb_sz):
    img_sz = img.shape
    I = np.zeros((img_emb_sz[0], img_emb_sz[1], 3))
    I[::] = np.nan
    I[img_emb_sz[0]//4:img_emb_sz[0]//4+img_sz[0], img_emb_sz[1]//4:img_emb_sz[1]//4+img_sz[1], :] = img
    return I / 255.


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    bottom_line = np.array([[0,0,1]])
    T_new = np.concatenate((T, bottom_line), axis=0)
    T_new = inv(T_new)
    return T_new[0:2, :]


def get_centered_transform(T, dx, dy):
    # "dx" should be width/2, "dy" should be height/2
    tx_tilde = -(dx*T[0][0]) + (dy*T[1][0]) + dx + T[0][2]
    ty_tilde = -(dx*T[1][0]) - (dy*T[0][0]) + dy + T[1][2]
    T_new = T.copy()
    T_new[0][2] = tx_tilde
    T_new[1][2] = ty_tilde
    return T_new


if __name__ == '__main__':
    main_STN()


