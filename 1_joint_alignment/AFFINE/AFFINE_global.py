import config
import numpy as np
from utils.image_warping import warp_image
import cv2
import matplotlib.pyplot as plt
from utils.Plots import open_figure, PlotImages


def get_global_AFFINE():
    imgs_big_embd = np.load('../data/imgs_big_embd.npy')
    SE_trans = np.load('../data/SE_trans.npy')
    AFFINE_residual = np.load('../data/AFFINE_residual.npy')
    img_embd_sz_arr = np.load('../data/img_embd_sz.npy')
    img_big_embd_sz_arr = np.load('../data/img_big_emb_sz.npy')

    #print('imgs_big_embd: ', imgs_big_embd.shape)
    #print('SE_trans: ', SE_trans.shape)
    #print('AFFINE_residual: ', AFFINE_residual.shape)

    img_big_emb_sz = (int(img_big_embd_sz_arr[0]),int(img_big_embd_sz_arr[1]),3)
    img_emb_sz = (int(img_embd_sz_arr[0]),int(img_embd_sz_arr[1]),3)

    # Prepare final transformations:
    final_trans = []
    imgs_trans_all = []
    for i, I in enumerate(imgs_big_embd):
        T_Final = concat_trans(SE_trans[i], np.reshape(AFFINE_residual[i], (2, 3)))
        I_warped, d = warp_image(I, T_Final, cv2.INTER_CUBIC)
        I_warped = embed_to_normal_sz_image(I_warped, img_emb_sz, img_big_emb_sz)
        I_warped = np.abs(I_warped / np.nanmax(I_warped))
        imgs_trans_all.append(I_warped)
        final_trans.append(T_Final)


    # build panoramic image with final transformations:
    # print('\nBuild panorama...')
    # panoramic_img = np.nanmedian(imgs_trans_all, axis=0)  # nanmean
    # fig1 = open_figure(1, 'Panoramic Image', (3, 2))
    # PlotImages(1, 1, 1, 1, [panoramic_img], [''], 'gray', axis=False, colorbar=False)
    # fig1.savefig('AFFINE/affine_alignment_results/Panorama_AFFINE_final.png', dpi=1000)
    # plt.show()

    # Save final transformations:
    final_trans = np.array(final_trans)
    np.save("../data/final_AFFINE_trans.npy", final_trans)


# Get two transformations in shape 2x3
# Return the composed trasnformation in shape 2x3
def concat_trans(T_SE, T_AFFINE0):
    bottom = np.expand_dims(np.array([0, 0, 1]), axis=0)
    T_SE = np.concatenate((T_SE, bottom), axis=0)
    T_AFFINE0 = np.concatenate((T_AFFINE0, bottom), axis=0)
    T_final = T_SE @ T_AFFINE0
    return T_final[0:2, :]


def embed_to_normal_sz_image(img, img_emb_sz, img_big_emb_sz):
    st_y = (img_big_emb_sz[0]-img_emb_sz[0])//2
    st_x = (img_big_emb_sz[1]-img_emb_sz[1])//2
    return img[st_y:st_y+img_emb_sz[0], st_x:st_x+img_emb_sz[1], :]