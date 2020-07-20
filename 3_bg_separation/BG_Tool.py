import sys
sys.path.append('..')

import os
import numpy as np
from utils.image_warping import warp_image
import cv2
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import matplotlib
from utils.Plots import open_figure, PlotImages
from data_provider import DataProvider
from Prediction.theta_prediction import run_theta_regressor
from Projection.FC_by_SIFT import get_FC_by_SIFT
from scipy.linalg import expm
import shutil
import config


class BG_Tool:

    def __init__(self):
        self.mypath = config.paths['my_path']
        self.data_path = self.mypath + 'data/'

        if config.bg_tool['use_gt_theta']:
            self.gt_trans = np.load(self.data_path + 'final_AFFINE_trans.npy')
            print('ground-truth transformations shape: ',self.gt_trans.shape)

        # img info:
        self.img_sz = config.images['img_sz']
        img_embd_sz_arr = np.load(self.data_path + 'img_embd_sz.npy')
        img_big_emb_sz_arr = np.load(self.data_path + 'img_big_emb_sz.npy')
        self.img_emb_sz = (int(img_embd_sz_arr[0]), int(img_embd_sz_arr[1]), 3)
        self.img_big_emb_sz = (int(img_big_emb_sz_arr[0]), int(img_big_emb_sz_arr[1]), 3)
        self.st_idx_y = (self.img_emb_sz[0] - self.img_sz[0]) // 2
        self.st_idx_x = (self.img_emb_sz[1] - self.img_sz[1]) // 2

        # pols info:
        self.window_sz = config.pols['window_sz']
        self.shift_sz = config.pols['shift_sz']
        self.is_zeromean = True
        if config.pols['method_type'] == 'PRPCA':
            self.is_zeromean = False

        # projection info:
        self.overlap_percent = config.bg_tool['overlap_percent']
        self.gap_refine = config.bg_tool['gap_refine']
        self.is_global_model = config.bg_tool['is_global_model']

        # prepare output folders:
        self.output_bg_path = self.mypath + 'output/bg'
        self.output_fg_path = self.mypath + 'output/fg'
        self.output_img_path = self.mypath + 'output/img'
        makedir(self.output_bg_path)
        makedir(self.output_fg_path)
        makedir(self.output_img_path)
        makedir(self.mypath + '3_bg_separation/test_imgs_alignment/')
        self.fg_imgs = None
        self.bg_imgs = None
        self.img_orig = None

        # required data structures:
        self.M = None  # number of test images
        self.idx = None # idx list of test images to process
        self.x_mean = None
        self.V = None
        self.fg_err = 0.05
        self.x_warped_overlap = None
        self.x_theta = None
        self.x_theta_inv = None
        self.x_theta_refine = None
        self.BG = None
        self.mean_err = None
        self.test_imgs = None
        self.test_trans = None
        self.test_trans_refine = None
        self.test_imgs_warped = None
        self.test_trans_all = None

    def load_test_images(self):
        data = DataProvider()
        self.idx = data.idx
        self.test_imgs = data.test_imgs

        # prepare arrays after knowing the exact number of test frames:
        self.M = self.test_imgs.shape[0]  # number of test images

        self.test_trans_all = np.zeros((self.M,2,3))  # predicted /ground-truth transformations for the tests images
        self.test_trans = np.zeros((self.M,2,3))  # only relevant transformations from self.test_trans_all
        self.test_trans_refine = np.zeros((self.M, 3, 3))  # refined transformations: affine + homography

        self.test_imgs_warped = np.zeros((self.M, self.img_emb_sz[0], self.img_emb_sz[1], self.img_emb_sz[2]))
        self.fg_imgs = np.zeros((self.M, self.img_sz[0], self.img_sz[1], self.img_sz[2]))
        self.bg_imgs = np.zeros((self.M, self.img_sz[0], self.img_sz[1], self.img_sz[2]))
        self.img_orig = np.zeros((self.M, self.img_sz[0], self.img_sz[1], self.img_sz[2]))
        self.mean_err = np.zeros((self.M, self.img_sz[0], self.img_sz[1]))

    def get_theta_for_test_imgs(self):
        if not config.bg_tool['use_gt_theta']:
            # Get predictions from net:
            preds = run_theta_regressor(self.test_imgs.copy(), self.mypath)
            self.test_trans_all = np.reshape(preds,self.test_trans.shape)  # should be in the shape of self.test_trans
        else:
            # Get ground-truth theta:
            self.test_trans_all = self.gt_trans[self.idx, ...]

    def prepare_test_image_transformations_refined(self):
        print('\nPrepare all transformed test images (with refined theta)..')
        tic = time.time()
        test_imgs_warped_before_refine = []
        cnt = 0

        panoramic_img = np.load(self.data_path + 'panoramic_img.npy')

        ref_image = np.zeros((self.img_emb_sz[0], self.img_emb_sz[1], self.img_emb_sz[2]))

        for i, I in enumerate(self.test_imgs):
            print('prepare image: ', self.idx[i])
            I = self.embed_to_big_image(I)
            if not config.bg_tool['use_gt_theta']:
                T_ = convert_to_expm(self.test_trans_all[i].ravel())
            else:
                T_ = self.test_trans_all[i]

            if config.bg_tool['only_refine']:
                self.gap_refine = config.bg_tool['gap_refine']
                T_ = np.eye(3)[0:2,0:3]
                T_[0][2] = -(self.img_emb_sz[1]//2 - self.img_sz[1]//2)
                T_[1][2] = -(self.img_emb_sz[0]//2 - self.img_sz[0]//2)

            T = np.reshape(T_, (2, 3))

            height, width, _ = I.shape
            x_warped, _ = warp_image(I, T, cv2.INTER_CUBIC)
            test_imgs_warped_before_refine.append(x_warped)

            # ------ Refine theta: use SIFT to warp x_warped towards X_mean_warped:
            I_warped_tmp = self.embed_to_normal_sz_image(x_warped.copy())

            start_x, start_y, end_x, end_y = self.get_enclosing_rectangle(I_warped_tmp.copy())
            x0, y0, x1, y1 = self.add_refine_gap_to_enclosing_rectangle(start_x, start_y, end_x, end_y)
            ref_image.fill(np.nan)
            ref_image[y0:y1, x0:x1, :] = panoramic_img[y0:y1, x0:x1, :]

            # refine the transformed test image:
            if self.is_global_model:
                H = np.eye(3)
            else:
                H = self.get_relative_trans(I_warped_tmp, ref_image)

            height,width,_ = I_warped_tmp.shape

            if H is not None:
                x_warped_new = cv2.warpPerspective(I_warped_tmp,H,(width,height),flags=cv2.INTER_CUBIC)
                x_warped_new = np.abs(x_warped_new / np.nanmax(x_warped_new))
                self.test_imgs_warped[cnt] = x_warped_new
                self.test_trans_refine[cnt] = H
                self.test_trans[cnt] = T

            cnt = cnt + 1

        self.test_imgs_warped = self.test_imgs_warped[:cnt]
        self.test_trans_refine = self.test_trans_refine[:cnt]
        self.test_trans = self.test_trans[:cnt]
        self.M = cnt

        toc = time.time()
        print('done preparing transformed images: ', toc - tic)

        panoramic = np.nanmedian(self.test_imgs_warped,axis=0)
        fig4 = open_figure(4,'Panoramic Image',(3,2))
        PlotImages(4,1,1,1,[panoramic],[''],'gray',axis=False,colorbar=False)
        plt.show()
        fig4.savefig(self.mypath + '3_bg_separation/test_imgs_alignment/test_imgs_panorama.png',dpi=1000)

        # panoramic2 = np.nanmedian(test_imgs_warped_before_refine,axis=0)
        # fig5 = open_figure(5,'Panoramic Image',(3,2))
        # PlotImages(5,1,1,1,[panoramic2],[''],'gray',axis=False,colorbar=False)
        # plt.show()
        # fig5.savefig(self.mypath + '3_bg_separation/test_imgs_alignment/test_imgs_before_refine_panorama.png', dpi=1000)

    def run_bg_model(self):
        print('\nRun bg model..')
        tic = time.time()

        # Save all backgrounds from overlapped images:
        self.BG = np.zeros((1000, self.img_sz[0], self.img_sz[1], self.img_sz[2]))

        # Run on all test images (for debug: it's the same):
        for i in range(self.M):
            tic_ = time.time()
            x_warped = self.test_imgs_warped[i]
            self.x_theta = self.test_trans[i]
            self.x_theta_inv = invert_T(self.x_theta)
            self.x_theta_refine = self.test_trans_refine[i]
            self.BG.fill(np.nan)
            cnt = 0

            # run on overlapped subspaces:
            start_x, start_y, end_x, end_y = self.get_enclosing_rectangle(x_warped.copy())
            for st_y in range(start_y, end_y, self.shift_sz):
                for st_x in range(start_x, end_x, self.shift_sz):
                    x_mean_filepath = self.data_path + 'subspaces/mean_' + str(st_y) + '_' + str(st_x) + '.npy'
                    V_filepath = self.data_path + 'subspaces/V_' + str(st_y) + '_' + str(st_x) + '.npy'
                    # if st_y <= end_y - self.window_sz[0] and st_x <= end_x - self.window_sz[1] and os.path.isfile(x_mean_filepath):
                    if os.path.isfile(x_mean_filepath):

                        # ------ Get V and X_mean of the overlapped subspace:
                        self.x_mean = np.load(x_mean_filepath)
                        self.V = np.load(V_filepath)
                        # plt.imshow(np.reshape(self.x_mean, (self.window_sz[0], self.window_sz[1], 3)))
                        # plt.show()

                        # ------ Get the pixels of x that overlap with V and X_mean:
                        x_warped_patch = np.expand_dims(x_warped[st_y:st_y+self.window_sz[0], st_x:st_x+self.window_sz[1], :].ravel(), axis=1)
                        mask = np.isnan(self.x_mean)
                        self.x_warped_overlap = np.where(mask, self.x_mean, x_warped_patch)
                        # move to next subspace if the overlap percentage is too low:
                        overlapped_pixels = len(self.x_warped_overlap[np.logical_not(np.isnan(self.x_warped_overlap))])
                        if overlapped_pixels > self.overlap_percent * self.window_sz[0]*self.window_sz[1]*3:
                            # ------ Project x_warped_overlap on V: (x_warped_overlap may include nans)
                            bg_r = self.project_x()

                            # ------ Put the bg in an image-embeded shape:
                            bg_emb = self.embed_bg(bg_r, st_x, st_y)

                            # ------ Return the bg to the original location of x:
                            bg = self.unwarp_bg(bg_emb)
                            self.BG[cnt] = bg
                            cnt = cnt + 1

            local_bg = np.nanmean(self.BG[:cnt, ...], axis=0)
            img = self.test_imgs[i]    #self.unwarp_original_image(x_warped)

            fg_binary, mean_err_per_pixel = self.compute_fg(local_bg, img)

            self.save_bg_fg(img, local_bg, fg_binary, mean_err_per_pixel, i)

            toc_ = time.time()
            print('Done local bg for image ', self.idx[i], ': ',  toc_ - tic_, ' using ', cnt, ' overlapped subspaces')

        toc = time.time()
        print('done bg computation for all images: ', toc - tic)

    def get_relative_trans(self, x_warped, x_mean_warped):
        H = get_FC_by_SIFT(x_warped, x_mean_warped)
        return H

    def get_enclosing_rectangle(self, img):

        if self.is_global_model:
            return 0, 0, self.img_emb_sz[0], self.img_emb_sz[1]

        # img of size: self.img_sz
        img = img[:, :, 0]
        np.nan_to_num(img, copy=False)

        x_axis = np.where(np.sum(img, axis=0) > 0, 1, 0)
        y_axis = np.where(np.sum(img, axis=1) > 0, 1, 0)

        x_axis_idx = np.where(x_axis == x_axis.max())
        y_axis_idx = np.where(y_axis == y_axis.max())

        start_x = np.min(x_axis_idx)
        start_y = np.min(y_axis_idx)
        end_x = np.max(x_axis_idx)
        end_y = np.max(y_axis_idx)

        # change the starting indices to fit the shift size:
        start_x = start_x - (start_x % self.shift_sz)
        start_y = start_y - (start_y % self.shift_sz)
        end_x = end_x - (end_x % self.shift_sz)
        end_y = end_y - (end_y % self.shift_sz)

        return start_x, start_y, end_x, end_y

    def add_refine_gap_to_enclosing_rectangle(self, start_x, start_y, end_x, end_y):
        if start_y - self.gap_refine >= 0:
            y_0 = start_y - self.gap_refine
        else:
            y_0 = 0

        if end_y + self.gap_refine < self.img_emb_sz[0]:
            y_1 = end_y + self.gap_refine
        else:
            y_1 = self.img_emb_sz[0]

        if start_x - self.gap_refine >= 0:
            x_0 = start_x - self.gap_refine
        else:
            x_0 = 0

        if end_x + self.gap_refine < self.img_emb_sz[1]:
            x_1 = end_x + self.gap_refine
        else:
            x_1 = self.img_emb_sz[1]

        return x_0, y_0, x_1, y_1

    def project_x(self):
        # handle missing data:
        if self.is_zeromean:
            x_zeromean = self.x_warped_overlap - self.x_mean
        else:
            x_zeromean = self.x_warped_overlap

        w_x = np.logical_not(np.isnan(self.x_warped_overlap))
        x_zeromean_tag = x_zeromean[w_x]
        w_x = np.squeeze(w_x)
        self.x_mean = np.squeeze(self.x_mean)
        U = self.V[w_x]
        A = U.T.dot(U)
        B = U.T.dot(x_zeromean_tag)
        alpha = self.solve_wls(A, B)
        bg_r = self.V @ alpha

        if self.is_zeromean:
            bg_r = bg_r + self.x_mean

        bg_r = np.abs(bg_r / np.nanmax(bg_r))
        return bg_r

    def solve_wls(self, A, B):
        alpha = np.linalg.lstsq(A, B, rcond=None)[0]
        return alpha

    def embed_bg(self, bg_r, st_x, st_y):
        bg = np.reshape(bg_r, (self.window_sz[0], self.window_sz[1], 3))
        bg_emb = np.full(self.img_emb_sz, np.nan)
        bg_emb[st_y:st_y+self.window_sz[0], st_x:st_x+self.window_sz[1], :] = bg
        return bg_emb

    def unwarp_bg(self,bg_emb):
        bg_emb_ = bg_emb
        height, width, _ = bg_emb_.shape
        bg_tmp = cv2.warpPerspective(bg_emb_, inv(self.x_theta_refine), (width, height), flags=cv2.INTER_CUBIC)
        bg, _ = warp_image(bg_tmp, self.x_theta_inv, cv2.INTER_CUBIC)
        bg = bg[self.st_idx_y:self.st_idx_y + self.img_sz[0], self.st_idx_x:self.st_idx_x + self.img_sz[1], :]
        bg = np.abs(bg / np.nanmax(bg))
        return bg

    def unwarp_original_image(self, x_warped):
        x_warped_ = x_warped
        height, width, _ = x_warped_.shape
        x_unwarped_tmp = cv2.warpPerspective(x_warped_, inv(self.x_theta_refine), (width, height), flags=cv2.INTER_CUBIC)
        x_unwarped, _ = warp_image(x_unwarped_tmp, self.x_theta_inv, cv2.INTER_CUBIC)
        x_unwarped = x_unwarped[self.st_idx_y:self.st_idx_y + self.img_sz[0], self.st_idx_x:self.st_idx_x + self.img_sz[1], :]
        x_unwarped = np.abs(x_unwarped / np.nanmax(x_unwarped))
        return x_unwarped

    def compute_fg(self, final_bg, x_unwarped):
        fg_tmp = np.square(x_unwarped - final_bg)
        mse_per_pixel = np.nanmean(fg_tmp, axis=2)
        mse_per_pixel = np.nan_to_num(mse_per_pixel)

        self.fg_err = np.nanpercentile(mse_per_pixel, 96)
        self.fg_err = 0.05
        new_fg = np.zeros((self.img_sz[0], self.img_sz[1], self.img_sz[2]))
        new_fg[mse_per_pixel > self.fg_err] = 1

        # final_fg = self.test_imgs[i] - final_bg
        # final_fg = final_fg / np.nanmax(final_fg)

        return new_fg, mse_per_pixel

    def save_bg_fg(self, img_orig, final_bg, fg_binary, mean_err_per_pixel, i):
        self.img_orig[i] = img_orig
        self.bg_imgs[i] = final_bg
        self.mean_err[i] = mean_err_per_pixel

        # Save all images:
        matplotlib.image.imsave(self.output_bg_path + '/f_' + str(self.idx[i]) + '.png', final_bg)
        matplotlib.image.imsave(self.output_fg_path + '/f_' + str(self.idx[i]) + '.png', mean_err_per_pixel, cmap='gray')
        matplotlib.image.imsave(self.output_img_path + '/f_' + str(self.idx[i]) + '.png', img_orig)

    def create_bg_fg_video(self):
        print('\nCreate videos from bg, fg, and original test images:')
        self.create_video(self.output_bg_path, 'bg_video')
        self.create_video(self.output_fg_path, 'fg_video')
        self.create_video(self.output_img_path, 'imgs_video')

    def create_video(self, folder_name, video_name):
        image_folder = folder_name
        video_name = video_name + '.avi'

        images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        h, w, _ = frame.shape

        video = cv2.VideoWriter(video_name, 0, 20, (w, h))

        # for image in images:
        for i in range(len(images)):
            video.write(cv2.imread(os.path.join(image_folder, 'f_' + str(i) + '.png')))

        # cv2.destroyAllWindows()
        video.release()

    def embed_to_normal_sz_image(self, img):
        st_y = (self.img_big_emb_sz[0] - self.img_emb_sz[0]) // 2
        st_x = (self.img_big_emb_sz[1] - self.img_emb_sz[1]) // 2
        if len(img.shape) > 3:
            return img[st_y:st_y + self.img_emb_sz[0], st_x:st_x + self.img_emb_sz[1], :, :]
        else:
            return img[st_y:st_y + self.img_emb_sz[0], st_x:st_x + self.img_emb_sz[1], :]

    def embed_to_big_image(self, img):
        I = np.zeros((self.img_big_emb_sz[0], self.img_big_emb_sz[1], 3))
        I[::] = np.nan
        start_idx_y = ((self.img_big_emb_sz[0] - self.img_emb_sz[0]) // 2) + (self.img_emb_sz[0] - self.img_sz[0]) // 2
        start_idx_x = ((self.img_big_emb_sz[1] - self.img_emb_sz[1]) // 2) + (self.img_emb_sz[1] - self.img_sz[1]) // 2
        I[start_idx_y:start_idx_y + self.img_sz[0], start_idx_x:start_idx_x + self.img_sz[1], :] = img
        return np.abs(I / np.nanmax(I))

    def embed_warped_image_to_big_image(self, img):
        I = np.zeros((self.img_big_emb_sz[0], self.img_big_emb_sz[1], 3))
        I[::] = np.nan
        start_idx_y = ((self.img_big_emb_sz[0] - self.img_emb_sz[0]) // 2)
        start_idx_x = ((self.img_big_emb_sz[1] - self.img_emb_sz[1]) // 2)
        I[start_idx_y:start_idx_y + self.img_emb_sz[0], start_idx_x:start_idx_x + self.img_emb_sz[1], :] = img
        return np.abs(I / np.nanmax(I))

    def get_centered_transform(self, T, pt):
        # "dx" should be width/2, "dy" should be height/2
        dx, dy = pt
        tx_tilde = -(dx*T[0][0]) + (dy*T[1][0]) + dx + T[0][2]
        ty_tilde = -(dx*T[1][0]) - (dy*T[0][0]) + dy + T[1][2]
        T_new = T.copy()
        T_new[0][2] = tx_tilde
        T_new[1][2] = ty_tilde
        return T_new

# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0, 0, 1]])
    T_new = np.concatenate((T, ones_line), axis=0)
    T_new = inv(T_new)
    return T_new[0:2, :]


# Get se transformation, in shape: (1,6)
# Return SE transformation, in shape (1,6)
def convert_to_expm(T):
    T = np.reshape(T, (2, 3))
    bottom = np.zeros((1, 3))
    T = np.concatenate((T, bottom), axis=0)
    T_exmp = expm(T)[0:2, :]
    return T_exmp.ravel()


def center_pts(p, q, img_sz):
    p[:,0] = p[:,0] - img_sz[1] / 2
    q[:,0] = q[:,0] - img_sz[1] / 2
    p[:,1] = p[:,1] - img_sz[0] / 2
    q[:,1] = q[:,1] - img_sz[0] / 2
    return p, q


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass

