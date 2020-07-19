import os
import shutil
import glob
import numpy as np
import cv2
from utils.video_to_frames import extractImages
import config


class DataProvider:

    def __init__(self):
        print('\nStart loading TEST data ...')
        input_path = '../input/test/'
        self.feed_path = 'tmp_data'
        makedir(self.feed_path)
        self.img_sz = config.images['img_sz']
        self.idx = None

        if config.bg_tool['data_type'] == "video":
            self.feed_size = extractImages(input_path + "video/" + config.bg_tool['video_name'], self.feed_path, (self.img_sz[1], self.img_sz[0]))
        else:
            list = glob.glob1(input_path + "images/", config.bg_tool['img_type'])
            list.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))
            self.feed_size = len(list)
            for i, filename in enumerate(list):
                shutil.copy(input_path + "images/" + filename, self.feed_path)
                os.rename(self.feed_path + '/' + filename, self.feed_path + '/frame' + str(i) + '.jpg')

        # number of test images to process:
        if config.bg_tool['which_test_frames'] == 'subsequence':
            self.idx = range(config.bg_tool['start_frame'], config.bg_tool['start_frame'] + config.bg_tool['num_of_frames'])
        elif config.bg_tool['which_test_frames'] == 'idx_list':
            self.idx = config.bg_tool['idx_list']
        else: # run on all test frames:
            self.idx = range(0, self.feed_size)

        # Read images to array:
        self.test_imgs = np.zeros((len(self.idx), self.img_sz[0], self.img_sz[1], self.img_sz[2]))
        j = 0
        for i in self.idx:
            img_str = self.feed_path + "/frame" + str(i) + ".jpg"
            I = cv2.imread(img_str)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            I = cv2.resize(I.copy(), (self.img_sz[1], self.img_sz[0]))
            self.test_imgs[j] = I   # np.abs(I / np.nanmax(I))
            j = j + 1

        print('Finished uploading test data:', self.test_imgs.shape)


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass

