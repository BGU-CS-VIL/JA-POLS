import numpy as np
import random
import config

class DataProvider:

    def __init__(self, nparray_path):
        print('Load data for STN ...')
        self.imgs_trans = np.load(nparray_path + 'SE_transformed_imgs_with_W.npy')
        img_embd_sz_arr = np.load(nparray_path + 'img_embd_sz.npy')

        self.feed_size = self.imgs_trans.shape[0]
        self.img_sz = (int(img_embd_sz_arr[0]), int(img_embd_sz_arr[1]), 4)
        self.cnt = 0

        idx_train = list(range(0, self.feed_size)) #random.sample(range(self.feed_size), self.feed_size)

        # self.batch_test = self.imgs_trans[idx_test, ...]
        self.batch_train = self.imgs_trans[idx_train, ...]

        print('Image size: ', self.img_sz)
        print('Finished uploading np array, Train data shape:', self.batch_train.shape, ', feed size: ', self.feed_size)

    def next_batch(self, batch_size, data_type):
        batch_img = None
        if data_type == 'train':

            if config.stn['ordered_batch']:  # (use ordered batch size)
                if self.cnt+batch_size <= self.feed_size:
                    idx = list(range(self.cnt, self.cnt+batch_size))
                    self.cnt = self.cnt+(batch_size//7)
                else:
                    idx = list(range(self.feed_size-batch_size, self.feed_size))
                    self.cnt = 0
                # END DEBUG (use ordered batch size)
            else:  # (use random batch size)
                idx = random.sample(range(len(self.batch_train)), batch_size)

            # print("idx: ", idx)
            batch_img = self.batch_train[idx, ...]
        elif data_type == 'test':
            batch_img = self.batch_train
        return batch_img
