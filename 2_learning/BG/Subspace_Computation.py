
import numpy as np
import time
from numpy.linalg import inv
from BG.rpca_candes_v1 import R_pca
from scipy.linalg import orth
import os
import shutil
import glob
import scipy.misc
from skimage import img_as_ubyte
import scipy
import matlab.engine
from scipy import io
import imageio


class Subspace_Computation:

    def __init__(self, mypath, data_path, N, k, trimming_percent, img_sz, method_type, imgs_trans_all, window_sz, shift_sz, overlap_percent, is_zeromean):
        self.N = N
        self.img_sz = img_sz
        self.k = k
        self.trimming_percent = trimming_percent
        self.d_tilde = window_sz[0] * window_sz[1] * 3 # number of pixels in a the subspace
        self.method_type = method_type
        self.imgs_trans_all = imgs_trans_all
        self.window_sz = window_sz
        self.shift_sz = shift_sz
        self.mypath = mypath
        self.data_path = data_path
        self.overlap_percent = overlap_percent
        self.is_zeromean = is_zeromean

        self.Y_full = np.zeros((self.N, self.d_tilde))
        self.W_full = np.zeros((self.N, self.d_tilde))
        self.Y = None
        self.W = None
        self.X_mean = None
        self.V = None

    def get_overlapped_imgs(self, st_y, st_x):
        tic = time.time()
        print('prepare subspace data..')

        self.Y_full.fill(0)
        self.W_full.fill(0)
        cnt = 0

        for i in range(0, self.N):
            patch = self.imgs_trans_all[i, st_y:st_y + self.window_sz[0], st_x:st_x + self.window_sz[1], :].ravel()
            patch_overlapped_pixels = patch[np.logical_not(np.isnan(patch))]  # check if the black area is nans or zeros
            if len(patch_overlapped_pixels) > self.overlap_percent * self.d_tilde:
                self.Y_full[cnt, :] = patch
                # Create a mask for the missing data (put 1 in non-nan pixels and 0 otherwise)
                W_tmp = self.W_full[cnt]
                np.copyto(dst=W_tmp, src=patch)
                W_tmp[np.logical_not(np.isnan(W_tmp))] = 1
                W_tmp[np.isnan(W_tmp)] = 0
                cnt = cnt + 1

        if cnt > 0:
            if cnt > 500 and self.overlap_percent != 0:
                cnt = 500

            self.Y = self.Y_full[:cnt]
            self.W = self.W_full[:cnt]
            n = self.Y.shape[0]
            self.Y = np.transpose(np.reshape(self.Y, (n, -1)))
            self.W = np.transpose(np.reshape(self.W, (n, -1)))
            print('data size: ', cnt)

            if self.method_type == 'PRPCA':
                self.Y = np.nan_to_num(self.Y)
            else:
                # Start with mean-data imputation:
                values = np.nanmean(self.Y, axis=1)  # nanmedian
                values = np.nan_to_num(values)
                values = np.repeat(values[:, np.newaxis], n, axis=1)
                self.Y = np.where(self.W, self.Y, values)

        toc = time.time()
        print('done preparing subspace data: ', toc - tic)
        # Y and W are dxN

        return cnt

    def learn_subspace(self):
        X_zeromean, self.X_mean = normalize_data(self.Y)
        if not self.is_zeromean:
            X_zeromean = self.Y

        if self.method_type == 'PCA':
            self.V = pca(X_zeromean, self.k)  # V is dxk

        elif self.method_type == 'PRPCA':
            PRPCA_path = self.mypath + '2_learning/BG/PRPCA/'

            X_zeromean_mat = np.reshape(X_zeromean, (self.window_sz[0], self.window_sz[1], 3, -1))
            X_zeromean_mat = np.reshape(X_zeromean_mat, (X_zeromean.shape[0], -1), order='F')
            W_zeromean_mat = np.reshape(self.W, (self.window_sz[0], self.window_sz[1], 3, -1))
            W_zeromean_mat = np.reshape(W_zeromean_mat, (self.W.shape[0], -1), order='F')

            # Save X_zeromean in a mat file
            io.savemat(PRPCA_path + 'x.mat', {'x': X_zeromean_mat})
            io.savemat(PRPCA_path + 'w.mat', {'w': W_zeromean_mat})

            # call matlab code that runs on the mat file:
            run_prpca_in_matlab(PRPCA_path)

            # Load L:
            L = scipy.io.loadmat(PRPCA_path + 'L.mat')['Lreg']

            L = np.reshape(L, (X_zeromean.shape[0], -1))
            self.V = orth(L)  # V is dxk, where k is the effective rank of the computed L
            print('V rank: ', self.V.shape[1])

        elif self.method_type == 'TGA':
            TGA_path = self.mypath + '2_learning/BG/TGA-PCA'
            makedir(TGA_path + '/subspaces')
            makedir(TGA_path + '/movie/starwars_000')

            # store self.Y as images (.jpg) on disk:
            for i in range(self.Y.shape[1]):
                img = np.reshape(self.Y[:, i], (self.window_sz[0], self.window_sz[1], 3))
                imageio.imwrite(TGA_path + '/movie/starwars_000/frame' + str(i) + '.jpg', img_as_ubyte(img))

            # Run TGA using shell command:
            # example: './TGA-PCA/build/GrassmannAveragesPCA_trimmed_ga_movie_runner [number of frames] [k] [trimming percentage] [path to TGA folder]'
            os.system('./BG/TGA-PCA/build/GrassmannAveragesPCA_trimmed_ga_movie_runner ' + str(self.Y.shape[1]) + ' ' + str(self.k) + ' ' + str(self.trimming_percent) + ' \"' + str(TGA_path) + "\"")
            print('finish running TGA.')

            self.V, X_mean = prepare_subspaces_soren(self.d_tilde, TGA_path + '/subspaces')

        elif self.method_type == 'RPCA-CANDES':
            rpca = R_pca(X_zeromean)
            L, S = rpca.fit(tol=1E-5, max_iter=1000, iter_print=10)  # V is dxN
            self.V = orth(L)  # V is dxk, where k is the effective rank of the computed L


    def save_subspace(self, st_y, st_x):
        np.save(self.data_path + 'subspaces/mean_' + str(st_y) + '_' + str(st_x) + '.npy', self.X_mean)
        np.save(self.data_path + 'subspaces/V_' + str(st_y) + '_' + str(st_x) + '.npy', self.V)


def normalize_data(X):
    X_mean = np.mean(X, axis=1)[:, np.newaxis]  # normalize X
    X_zeromean = X - X_mean
    return X_zeromean, X_mean


# Prepare subspaces V and X_mean from Soren's code:
def prepare_subspaces_soren(d, soren_path):
    # Read X_mean:
    X_mean = read_all_text_to_float_array(soren_path + '/mean_vector.txt')
    X_mean = np.expand_dims(X_mean, axis=1)

    # Read V (pca tga):
    list = sorted(glob.glob1(soren_path, 'vector_subspace*'))
    V_pca_tga = np.zeros((d, len(list)))
    for i, filename in enumerate(list):
        path_v = soren_path + '/' + filename
        V_pca_tga[:, i] = read_all_text_to_float_array(path_v)

    return V_pca_tga, X_mean


def read_all_text_to_float_array(path):
    f = open(path, "r")
    tokens = []
    for line in f:
        tokens = tokens + line.split(' ')[:-1]

    arr_ = np.array(tokens)
    return arr_.astype(np.float)


def pca(X, k): # X: dxN
    V, S, U = np.linalg.svd(X, full_matrices=False)  # compute PCA
    Vk = V[:, :k]
    return Vk


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0, 0, 1]])
    T_new = np.concatenate((T, ones_line), axis=0)
    T_new = inv(T_new)
    return T_new[0:2, :]


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass


def run_prpca_in_matlab(matlab_path):
    eng = matlab.engine.start_matlab()
    eng.addpath(r'' + matlab_path, nargout=0)
    eng.main(nargout=0)
    eng.quit()
