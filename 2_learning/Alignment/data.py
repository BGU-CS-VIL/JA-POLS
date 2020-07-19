import numpy as np
import platform
import random
import torch
from torchvision import transforms
from torch.utils import data
from scipy.linalg import logm, expm
from utils.image_warping import warp_image


class DataSetRegress(data.Dataset):
    def __init__(self, imgs, embedded_imgs, thetas, normalizers):
        self.imgs = imgs
        self.embedded_imgs = embedded_imgs
        self.thetas = thetas
        if normalizers != None:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(normalizers['mean'], normalizers['std'])])
        else:
            self.transform = [transforms.ToTensor()]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        image = self.transform(self.imgs[index])
        lbl = self.thetas[index]
        return image, lbl


class DataSetRegressTest(data.Dataset):
    def __init__(self, imgs, normalizers):
        self.imgs = imgs
        if normalizers != None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalizers['mean'], normalizers['std'])])
        else:
            self.transform = [transforms.ToTensor()]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        image = self.transform(self.imgs[index])
        return image, np.zeros((1, 6))


def convert_to_logm(T):
    """
    Get SE transformation, in shape: (1,6) \n
    Return SE transformation, in shape (1,6)
    """
    T = np.reshape(T, (2, 3))
    bottom = np.zeros((1, 3))
    bottom[0][2] = 1
    T = np.concatenate((T, bottom), axis=0)
    T_log = logm(T)[0:2, :]
    return T_log.ravel()


def split_train_test(ds_size, test_prcnt):
    test_size = int(ds_size*test_prcnt)
    idx_test = random.sample(range(ds_size), test_size)
    all_idx = random.sample(range(ds_size), ds_size)
    idx_train = [x for x in all_idx if x not in idx_test]

    return idx_train, idx_test


def get_datasets(data_dir, test_prcnt, augment, normalizers, aug_num_per_img=10, aug_std=1):
    print('Start loading data...\n')

    if platform.system() == 'Windows':
        imgs = str(data_dir)+'imgs.npy'
        imgs_embdings = str(data_dir)+'imgs_big_embd.npy'
        imgs_transf = str(data_dir)+'final_AFFINE_trans.npy'

    else:
        imgs = str(data_dir)+'imgs.npy'
        imgs_embdings = str(data_dir)+'imgs_big_embd.npy'
        imgs_transf = str(data_dir)+'final_AFFINE_trans.npy'

    # Load npy files:
    x = np.load(imgs)
    theta_gt_exp = np.load(imgs_transf)
    x_embedded = np.load(imgs_embdings)

    ds_size = x.shape[0]

    # Get split indices:
    idx_train, idx_test = split_train_test(ds_size, test_prcnt)

    train_imgs = x[idx_train]
    train_theta = theta_gt_exp[idx_train]
    train_x_embd = x_embedded[idx_train]

    print('Pre aug:')
    print(train_imgs.shape, train_theta.shape, train_x_embd.shape)
    test_imgs = x[idx_test]
    test_theta = theta_gt_exp[idx_test]
    test_x_embd = x_embedded[idx_test]

    if augment:
        train_imgs, train_x_embd, train_theta = augment_data(
            train_imgs, train_x_embd, train_theta, aug_num_per_img, std=aug_std)

    print('After aug:')
    print(train_imgs.shape, train_theta.shape, train_x_embd.shape)

    # Convert theta's to log scale:
    test_theta_log = np.zeros((test_theta.shape[0], 6))
    for i in range(test_theta.shape[0]):
        test_theta_log[i] = convert_to_logm(test_theta[i])

    train_theta_log = np.zeros((train_theta.shape[0], 6))
    for i in range(train_theta.shape[0]):
        train_theta_log[i] = convert_to_logm(train_theta[i])

    # In PyTorch, image data is expected to be in this format: [batch_size, channel, height, width].
    # train_imgs = train_imgs.transpose([0, 3, 1, 2])
    # test_imgs = test_imgs.transpose([0, 3, 1, 2])

    # Create datasets:
    train_ds = DataSetRegress(train_imgs, train_x_embd,
                              train_theta_log, normalizers)
    test_ds = DataSetRegress(test_imgs, test_x_embd,
                             test_theta_log, normalizers)

    print('Finished loading data\n')
    print('Images size: ', train_ds.imgs.shape[1:])
    print('Total images in train-set: ', train_ds.imgs.shape[0])
    print('Total images in test-set: ', test_ds.imgs.shape[0])
    # print('Embedded-image size: ', self.img_big_emd_sz)
    # print('train images sz: ', self.train_imgs.shape)
    # print('test images sz: ', self.test_imgs.shape)
    # print('train theta sz: ', self.train_theta.shape)
    # print('test theta sz: ', self.test_theta.shape)
    # print('test x_embd sz: ', self.test_x_embd.shape)

    return train_ds, test_ds


def get_datasets_for_test(test_imgs, normalizers):
    print('Preprocess data for net...\n')
    test_ds = DataSetRegressTest(test_imgs, normalizers)

    print('Finished loading test data\n')
    print('Images size: ', test_ds.imgs.shape[1:])
    print('Total images in test-set: ', test_ds.imgs.shape[0])
    return test_ds


def random_affine_T(deg=None, t_xy=None, just_SE=False, std=1):
    """
    Get a random affine transformation matrix
    Args:
    ----------
    - deg: To determine specific rotation angle in degrees
    - t_xy: To determine specific translation (t_x, t_y)

    Returns:
    ----------
    - Affine transformation matrix, shaped (2, 3)
    """
    # Initilize T to zeros and get a random translation vector
    T = np.zeros((2, 3))
    if t_xy == None:
        t_xy = np.random.randint(-20, 20, size=(2, ))

    # An SE transformation
    if just_SE:
        if deg == None:
            deg = np.random.randint(-10, 10)
        deg = np.deg2rad(deg)
        R_mat = np.array([[np.cos(deg), -np.sin(deg)], [np.sin(deg), np.cos(deg)]])
        T[:, 0:2] = R_mat
    # An Affine transformation
    else:
        theta = np.random.normal(size=6, scale=std).reshape((2,3))
        A = np.zeros((3,3))
        A[:-1] = theta
        T = expm(A)[:-1]

    T[:, 2] = t_xy
    return T


def compose_transform(T1, T2):
    """
    Get an equivalent SE transformation matrix T, \n
    from (x',y') = T2(T1(x,y)) = T(x,y)

    Args:
    ----------
    - T1: the first SE matrix, shaped (2, 3)
    - T2: the second SE matrix, shaped (2, 3)

    Returns:
    ----------
    - SE transformation matrix T, shaped (2, 3)
    """
    aux_vec = np.array([0, 0, 1]).reshape(1, 3)

    T1 = np.concatenate((T1, aux_vec), axis=0)
    T2 = np.concatenate((T2, aux_vec), axis=0)

    T1_inv = np.linalg.inv(T1)
    T = T1_inv@T2

    return T[0:2]


def augment_data(imgs, embbed_imgs, thetas, aug_num_per_img, std=1):

    aug_imgs = np.repeat(imgs, aug_num_per_img, axis=0)
    aug_thetas = np.repeat(thetas, aug_num_per_img, axis=0)
    aug_img_embbed = np.repeat(embbed_imgs, aug_num_per_img, axis=0)

    for j, (img, embbed_img, theta) in enumerate(zip(imgs, embbed_imgs, thetas)):
        for i in range(aug_num_per_img):
            T = random_affine_T(std=std)
            aug_imgs[j*aug_num_per_img+i], _ = warp_image(img, T)
            aug_img_embbed[j*aug_num_per_img+i], _ = warp_image(embbed_img, T)
            aug_thetas[j*aug_num_per_img+i] = compose_transform(T, theta)

    return aug_imgs, aug_img_embbed, aug_thetas


