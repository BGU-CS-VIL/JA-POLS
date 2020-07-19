
import numpy as np
from utils.image_warping import warp_image
from BG.Subspace_Computation import Subspace_Computation
import cv2
import gc
import os
import shutil
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils.Plots import open_figure, PlotImages
import config


class Piecewise_Subspace:

    def __init__(self):
        self.mypath = config.paths['my_path']
        self.data_path = self.mypath + 'data/'
        self.window_sz = config.pols['window_sz']
        self.shift_sz = config.pols['shift_sz']  # number of pixels between windows
        self.k = config.pols['k']
        self.method_type = config.pols['method_type']
        self.trimming_percent = config.pols['trimming_percent']
        self.overlap_percent = config.pols['overlap_percent']  # minimum % of overlapped pixels out of d_tilde, to consider an overlapped image (used in "get_overlapped_imgs")
        self.min_data_points = config.pols['min_data_points']  # minimum number of images to learn subspace from.
        self.is_zeromean = True
        if self.method_type == 'PRPCA':
            self.is_zeromean = False

        self.img_sz = config.images['img_sz']
        img_embd_sz_arr = np.load(self.data_path + 'img_embd_sz.npy')
        img_big_emb_sz_arr = np.load(self.data_path + 'img_big_emb_sz.npy')
        self.img_emb_sz = (int(img_embd_sz_arr[0]), int(img_embd_sz_arr[1]), 3)
        self.img_big_emb_sz = (int(img_big_emb_sz_arr[0]), int(img_big_emb_sz_arr[1]), 3)

        self.imgs = np.load(self.data_path + 'imgs.npy')
        self.imgs_big_embd = np.load(self.data_path + 'imgs_big_embd.npy')
        self.trans = np.load(self.data_path + 'final_AFFINE_trans.npy')

        self.N = self.imgs.shape[0]
        self.imgs_trans_all = np.zeros((self.N, self.img_emb_sz[0], self.img_emb_sz[1], self.img_sz[2]))

        makedir(self.data_path + 'subspaces/')

        print('img_sz: ',self.img_sz)
        print('img_emb_sz: ',self.img_emb_sz)
        print('img_big_emb_sz: ',self.img_big_emb_sz)
        print('imgs_big_embd: ',self.imgs_big_embd.shape)
        print('final_AFFINE_trans: ',self.trans.shape)

    def prepare_image_transformations(self):
        tic = time.time()
        print('\nPrepare all transformed images..')
        for i, I in enumerate(self.imgs_big_embd):
            I_warped, _ = warp_image(I, self.trans[i], cv2.INTER_CUBIC)
            I_warped = self.embed_to_normal_sz_image(I_warped)
            I_warped = np.abs(I_warped / np.nanmax(I_warped))
            self.imgs_trans_all[i] = I_warped

        panoramic_img = np.nanmean(self.imgs_trans_all, axis=0)  # nanmean
        np.save(self.data_path + 'panoramic_img.npy', panoramic_img)

        fig4 = open_figure(4,'Panoramic Image',(3,2))
        PlotImages(4,1,1,1,[panoramic_img],[''],'gray',axis=False,colorbar=False)
        plt.show()
        fig4.savefig(self.mypath + '2_learning/BG/alignment_mean/panorama.png', dpi=1000)

        toc = time.time()
        print('done images transformations: ', toc - tic)

    def run_PS_subspace(self):
        print('\nLearn local subspaces..')

        sbspace = Subspace_Computation(self.mypath, self.data_path, self.N, self.k, self.trimming_percent, self.img_sz, self.method_type, self.imgs_trans_all, self.window_sz, self.shift_sz, self.overlap_percent, self.is_zeromean)

        subspaces_num = np.zeros(1)
        for st_y in range(0, self.img_emb_sz[0], self.shift_sz):
            for st_x in range(0, self.img_emb_sz[1], self.shift_sz):
                if st_y <= self.img_emb_sz[0] - self.window_sz[0] and st_x <= self.img_emb_sz[1] - self.window_sz[1]:
                    if st_y == 160 and st_x < 200:
                        continue
                    print('compute subspace for idx: ', st_y, ', ', st_x)
                    num_of_datapoints = sbspace.get_overlapped_imgs(st_y, st_x)
                    if num_of_datapoints > self.min_data_points:
                        sbspace.learn_subspace()
                        sbspace.save_subspace(st_y, st_x)
                        subspaces_num[0] = subspaces_num[0] + 1
                        gc.collect()

        print('Finished computing ', subspaces_num, ' local subspaces.')

    def embed_to_normal_sz_image(self, img):
        st_y = (self.img_big_emb_sz[0] - self.img_emb_sz[0]) // 2
        st_x = (self.img_big_emb_sz[1] - self.img_emb_sz[1]) // 2
        return img[st_y:st_y + self.img_emb_sz[0], st_x:st_x + self.img_emb_sz[1], :]


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0, 0, 1]])
    T_new = np.concatenate((T, ones_line), axis=0)
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
