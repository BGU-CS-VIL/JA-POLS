
import os
import numpy as np
import matplotlib.pyplot as plt
from SE.Edge import Edge
from SE.FC_by_SIFT import get_FC_by_SIFT
from utils.Plots import open_figure, PlotImages
import cv2
from utils.image_warping import warp_image
from numpy.linalg import inv
import config


class GraphManager:

    def __init__(self, data):
        self.E = dict()
        self.V = dict()
        self.E_pts = dict()
        self.E_inliers_sz = dict()
        self.data = data
        self.create_E()
        self.create_V()

    def create_E(self):
        print("Create set of edges E ...")
        for i in range(self.data.feed_size-3):
            for j in range(3):
                edge = Edge(i, i + j + 1)
                self.E[edge] = np.zeros((2,3))

        i = self.data.feed_size - 3
        edge = Edge(i, i + 1)
        self.E[edge] = np.zeros((2,3))
        edge = Edge(i, i + 2)
        self.E[edge] = np.zeros((2,3))
        edge = Edge(i + 1, i + 2)
        self.E[edge] = np.zeros((2,3))

    def create_V(self):
        print("Create set of vertices V ...")
        for i in range(self.data.feed_size):
            self.V[i] = np.zeros((2, 3))

    def compute_relative_trans(self, path):
        print("Compute relative transformations for each edge in E ...")
        if os.path.exists(path + 'output_file'):
            os.remove(path + 'output_file')
        f = open(path + 'output_file', 'w+')
        pt = float(self.data.img_sz[1]) / 2, float(self.data.img_sz[0]) / 2
        for e in self.E:
            # ------------ TEST RANSAC POINTS
            p, q, w = get_FC_by_SIFT(self.data.imgs[e.src], self.data.imgs[e.dst])

            min_num_of_FC_points = 50   # if there are not enough FC points, don't provide this as a measurement to se-sync.

            if p.shape[0] <  min_num_of_FC_points:
                continue

            p, q = center_pts(p,q,self.data.img_sz)
            ransac_iterations = 200
            ransac_sbgrp_sz = round(0.1 * p.shape[0])
            idx = np.arange(p.shape[0])
            self.E_inliers_sz[e] = 0
            self.E_pts[e] = p.copy(), q.copy()
            n_pts = p.shape[0]
            epsilon = 1.1
            idx_best = (1,)
            # perform RANSAC iterations
            for it in range(ransac_iterations):
                # choose random sample of points:
                np.random.shuffle(idx)    # use instead: idx_sbgrp = random.sample(range(p.shape[0]), ransac_sbgrp_sz)
                idx_sbgrp = idx[:ransac_sbgrp_sz]
                p_sbgrp = p[idx_sbgrp,:]
                q_sbgrp = q[idx_sbgrp,:]
                w_sbgrp = w[idx_sbgrp,:]

                # Compute relative transformation:
                if len(w_sbgrp) > 1:
                    T_e = compute_LS_rigid_motion(p_sbgrp,q_sbgrp,w_sbgrp)
                    T_e = get_centered_transform(T_e, pt[0], pt[1])

                    # Apply T_e on q (all points):
                    ones_line = np.ones((1,n_pts))
                    xy_q_pts = np.concatenate(
                        (np.reshape(q[:n_pts,:1], (1,n_pts)), np.reshape(q[:n_pts,1:2],(1,n_pts)),ones_line),axis=0)
                    q_trans = T_e @ xy_q_pts
                    q_trans = q_trans[:2,].transpose()

                    # Classify inliers out of all points:
                    mse_arr = np.square(np.linalg.norm(p - q_trans, axis=1))
                    idx_inliers = np.where(mse_arr < epsilon)[0]

                    if len(idx_inliers) > len(idx_best):
                        idx_best = idx_inliers.copy()

            # Compute T_e based on idx_best:
            p_sbgrp = p[idx_best, :]
            q_sbgrp = q[idx_best, :]
            w_sbgrp = w[idx_best, :]
            # print(len(w_sbgrp), len(w))
            if len(w_sbgrp) < 0.7*len(w):
                self.E[e] = compute_LS_rigid_motion(p,q,w)
            else:
                self.E[e] = compute_LS_rigid_motion(p_sbgrp,q_sbgrp,w_sbgrp)

            self.E_inliers_sz[e] = len(w_sbgrp)

            # Add this measurement to the output file:
            pose = ' '.join([str(p) for p in np.reshape(self.E[e], (6, ))])
            tau = 0
            kappa = 0
            f.write('EDGE_SE2 ' + str(e.src) + ' ' + str(e.dst) + ' ' + pose + '\n')
            # ------------ END TEST RANSAC POINTS

        print("Finished.")

    def read_sesync_output(self, sesync_path):
        f = open(sesync_path, "r")
        for line in f:
            tokens = line[:-1].split(' ')
            tokens = [float(i) for i in tokens]
            v = int(tokens[0])
            self.V[v] = extract_optimal_pose(tokens[1:])

    def transform_images_globally(self, n_frames, nparray_path):
        print('\nTransform Images globally...')
        imgs_trans_all = []
        img_sz = self.data.img_sz

        # The transformations received from SE-Sync are: from the original image location (center) -> to the global location in the panorama.

        # find the middle frame to create a minimum-size panoramic domain:
        trans_x = []
        for v in self.V:
            if v in range(0, n_frames):
                trans_x.append(self.V[v].ravel())

        trans_x = np.array(trans_x)
        middle_frame = np.argsort(trans_x[:,2])[trans_x.shape[0] // 2]
        print('middle_frame: ', middle_frame)

        # prepare transformations from center:
        trans = []
        T0_inv = invert_T(self.V[middle_frame])
        for v in self.V:
            if v in range(0, n_frames):
                T = revert_to_T0(self.V[v], T0_inv)
                if np.isfinite(np.linalg.cond(T)):
                    T_SE = invert_T(T)
                    if T_SE is not None:
                        trans.append(T_SE.ravel())
                    else:
                        trans.append(np.zeros(2,3).ravel())
                else:
                    trans.append(np.zeros(2,3).ravel())

        trans = np.array(trans)

        # get extreme translations to get estimation for img_embd_sz:
        x_sz = (2 * (int(np.max(np.abs(trans[:,2]))))) + np.max(img_sz) + 50
        y_sz = (2 * (int(np.max(np.abs(trans[:,5]))))) + np.max(img_sz) + 50
        img_emb_sz = (y_sz, x_sz, 3)
        img_big_emb_sz = (y_sz+400, x_sz+400, 3)

        print('img_emb_sz: ', img_emb_sz)
        print('img_big_emb_sz: ', img_big_emb_sz)
        np.save(nparray_path + "img_embd_sz.npy", img_emb_sz)
        np.save(nparray_path + "img_big_emb_sz.npy", img_big_emb_sz)

        # transform images in embeded frames:
        for v in self.V:
            if v in range(0, n_frames):
                T_SE = np.reshape(trans[v], (2,3))

                # embed the image:
                I = embed_to_big_image(self.data.imgs[v], img_emb_sz, img_big_emb_sz)

                if np.sum(T_SE) != 0: # if the transformation is valid, use it.
                    I_Rt, d = warp_image(I, T_SE, cv2.INTER_CUBIC)
                    I_Rt = np.abs(I_Rt)
                    I_Rt = embed_to_normal_sz_image(I_Rt, img_emb_sz, img_big_emb_sz)
                    imgs_trans_all.append(I_Rt)
                    self.data.imgs_trans.append(I_Rt)
                    self.data.trans.append(T_SE)
                    self.data.imgs_big_embd.append(I)
                    self.data.imgs_relevant.append(np.abs(self.data.imgs[v] / np.nanmax(self.data.imgs[v])))

        # build panoramic image (Sesync results):
        print('\nBuild panorama...')
        panoramic_img = np.nanmedian(imgs_trans_all, axis=0) # nanmean
        fig1 = open_figure(1, 'Panoramic Image', (3, 2))
        PlotImages(1, 1, 1, 1, [panoramic_img], [''], 'gray', axis=False, colorbar=False)
        fig1.savefig(config.paths['my_path'] + '1_joint_alignment/SE/se_alignment_results/SE_Panorama.png', dpi=1000)
        plt.show()

    def prepare_data_for_STN(self, nparray_path):
        imgs_trans_with_W_ravel = []
        for i, I in enumerate(self.data.imgs_trans):
            W = np.reshape(I[:, :, 0], (I.shape[0], I.shape[1], 1))
            W = (np.isnan(W) < 1).astype(int) # replace non-Nan with 1, and Nan with 0
            I = np.nan_to_num(I)
            I_with_W = np.concatenate((I, W), axis=2)
            imgs_trans_with_W_ravel.append(I_with_W.ravel())

        imgs_trans_with_W_ravel = np.array(imgs_trans_with_W_ravel)
        self.data.img_sz = I_with_W.shape
        self.data.example_img = I_with_W
        self.data.imgs_big_embd = np.array(self.data.imgs_big_embd)
        self.data.imgs_relevant = np.array(self.data.imgs_relevant)

        print('\nSave numpy arrays to STN...')
        # save imgs_trans as numpy arrays for STN:
        np.save(nparray_path + "imgs.npy", self.data.imgs_relevant)
        np.save(nparray_path + "imgs_big_embd.npy", self.data.imgs_big_embd)
        np.save(nparray_path + "SE_transformed_imgs_with_W.npy", imgs_trans_with_W_ravel)
        np.save(nparray_path + "SE_trans.npy", self.data.trans)


def center_pts(p, q, img_sz):
    p[:,0] = p[:,0] - img_sz[1] / 2
    q[:,0] = q[:,0] - img_sz[1] / 2
    p[:,1] = p[:,1] - img_sz[0] / 2
    q[:,1] = q[:,1] - img_sz[0] / 2
    return p, q


# Input: image I
# Output: image I after replacing zeros with nan
def nan_if(I):
    I_new = np.where(I == 0.0, np.nan, I)
    return I_new


# p_i.shape = 2x1; q_i.shape = 2x1; w_i is a scalar
# Returns theta (relative transformation) in shape 2x3
def compute_LS_rigid_motion(p, q, w):
    d = 2

    # Compute t & R:

    p_bar = np.sum(w * p, 0) / np.sum(w)
    q_bar = np.sum(w * q, 0) / np.sum(w)

    X = p - p_bar
    Y = q - q_bar

    W = np.diag(np.squeeze(w))

    S = X.transpose() @ W @ Y

    U, Sigma, V = np.linalg.svd(S)

    H = np.eye(d)
    H[d-1, d-1] = np.linalg.det(V @ U.transpose())

    R = V @ H @ U.transpose()
    t = q_bar - R @ p_bar

    upper_matrix = np.concatenate((R, np.expand_dims(t, axis=1)), axis=1)
    bottom_matrix = np.expand_dims(np.array([0,0,1]), axis=0)
    theta = np.concatenate((upper_matrix, bottom_matrix), axis=0)

    theta = inv(theta)

    return theta[0:2, :]


def extract_optimal_pose(tokens):
    poses = np.asarray(tokens)
    # debug (using STN without smart initialization):
    # poses = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return np.reshape(poses, (2, 3))


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0,0,1]])
    T_new = np.concatenate((T, ones_line), axis=0)
    if is_invertible(T_new):
        T_new = inv(T_new)
        return T_new[0:2,:]
    else:
        print('not invertible: ', T_new)
        return None


# Input: 2 transformations T and T1, 2x3.
# Output: transformation ~T, 2x3
def revert_to_T0(T, T0_inv):
    bottom_line = np.array([[0, 0, 1]])
    T0_inv_ = np.concatenate((T0_inv, bottom_line), axis=0)
    T_ = np.concatenate((T, bottom_line), axis=0)
    T_tilde = T0_inv_ @ T_
    return T_tilde[0:2, :]


def embed_to_big_image(img, img_emb_sz, img_big_emb_sz):
    img_sz = img.shape
    I = np.zeros((img_big_emb_sz[0], img_big_emb_sz[1], 3))
    I[::] = np.nan
    start_idx_y = ((img_big_emb_sz[0]-img_emb_sz[0])//2)+(img_emb_sz[0]-img_sz[0])//2
    start_idx_x = ((img_big_emb_sz[1]-img_emb_sz[1])//2)+(img_emb_sz[1]-img_sz[1])//2
    I[start_idx_y:start_idx_y+img_sz[0], start_idx_x:start_idx_x+img_sz[1], :] = img
    return np.abs(I / np.nanmax(I))


def embed_to_normal_sz_image(img, img_emb_sz, img_big_emb_sz):
    st_y = (img_big_emb_sz[0]-img_emb_sz[0])//2
    st_x = (img_big_emb_sz[1]-img_emb_sz[1])//2
    return img[st_y:st_y+img_emb_sz[0], st_x:st_x+img_emb_sz[1], :]


def axis_ij(g=None):
    if g is None:
        g = plt.gca()
    bottom, top = g.get_ylim()
    if top>bottom:
        g.set_ylim(top, bottom)
    else:
        pass


def get_centered_transform(T, dx, dy):
    # "dx" should be width/2, "dy" should be height/2
    tx_tilde = -(dx*T[0][0]) + (dy*T[1][0]) + dx + T[0][2]
    ty_tilde = -(dx*T[1][0]) - (dy*T[0][0]) + dy + T[1][2]
    T_new = T.copy()
    T_new[0][2] = tx_tilde
    T_new[1][2] = ty_tilde
    return T_new


def MSE(X, Y):
    return np.square(X - Y).mean()


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
