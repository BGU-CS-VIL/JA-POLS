
import numpy as np
from numpy.linalg import inv
import cv2


# T: 2x3 affine matrix, which is the INVERSE map!
# pt: (x,y): the point in which the rotation will be around
# Assume that the rotation in T is around the middle of the image.
def warp_image(I, T, interpolation = cv2.INTER_CUBIC):
    T_new = T.copy()
    height, width, _ = I.shape

    map_x, map_y = create_mappings(T_new, height, width)

    # Interpolate image:
    I_warpped_remap = cv2.remap(I.copy(), map_x, map_y, interpolation)

    d={}
    d['map_x'] = map_x
    d['map_y'] = map_y
    return I_warpped_remap, d


def create_mappings(T, height, width):
    # Create grid os size height x width of values in the range of [-1,1]:
    ones_x = np.ones((height, 1))
    intervales_x = np.reshape(np.arange(-width/2, width/2, 1), (1, width)) # np.reshape(np.linspace(-1, 1, num=width), (1, width))
    x = ones_x @ intervales_x
    ones_y = np.ones((1, width))
    intervales_y = np.reshape(np.arange(-height/2, height/2, 1), (height, 1)) # np.reshape(np.linspace(-1, 1, num=height), (height, 1))
    y = intervales_y @ ones_y

    # Transform the grid:
    y_pts = np.reshape(y, (1, -1))
    x_pts = np.reshape(x, (1, -1))
    ones_line = np.ones((1, height*width))
    xy_pts = np.concatenate((x_pts, y_pts, ones_line), axis=0)
    xy_pts_trans = T @ xy_pts  # 2 x num_of_pixels

    # Normalize map_x, map_y to be in the range of [height, width]:
    map_x_sml = xy_pts_trans[0, :]
    map_y_sml = xy_pts_trans[1, :]
    map_x = ((map_x_sml/(width/2.0)) + 1.0) * (width/2.0)  #(map_x_sml + 1.0) * width / 2.0
    map_y = ((map_y_sml/(height/2.0)) + 1.0) * (height/2.0) #(map_y_sml + 1.0) * height / 2.0

    map_x = np.reshape(map_x, (height, width)).astype(np.float32)
    map_y = np.reshape(map_y, (height, width)).astype(np.float32)
    return map_x, map_y


# Input: transformation T, 2x3
# Output: transformation T^{-1}, 2x3
def invert_T(T):
    ones_line = np.array([[0,0,1]])
    T_new = np.concatenate((T, ones_line), axis=0)
    T_new = inv(T_new)
    return T_new[0:2, :]


def get_centered_transform(T, pt):
    # "dx" should be width/2, "dy" should be height/2
    dx, dy = pt
    tx_tilde = -(dx*T[0][0]) + (dy*T[1][0]) + dx + T[0][2]
    ty_tilde = -(dx*T[1][0]) - (dy*T[0][0]) + dy + T[1][2]
    T_new = T.copy()
    T_new[0][2] = tx_tilde
    T_new[1][2] = ty_tilde
    return T_new
