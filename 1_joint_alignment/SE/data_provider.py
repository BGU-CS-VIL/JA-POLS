import os
import shutil
import glob
import cv2
from utils.video_to_frames import extractImages
import config


class DataProvider:

    def __init__(self):

        # create the data folders
        makedir(config.paths['my_path'] + 'data/')
        makedir(config.paths['my_path'] + '1_joint_alignment/SE/se_alignment_results/')
        makedir(config.paths['my_path'] + '1_joint_alignment/STN/stn_alignment_results/')
        makedir(config.paths['my_path'] + '1_joint_alignment/STN/model/')
        makedir(config.paths['my_path'] + '1_joint_alignment/AFFINE/affine_alignment_results/')

        input_path = '../input/learning/'
        self.feed_path = "SE/tmp_data"
        makedir(self.feed_path)
        self.img_sz = config.images['img_sz']  # set "-1" in case we want the original img_sz.

        print('Start loading data ...')

        if config.se['data_type'] == "video":
            self.feed_size = extractImages(input_path + "video/" + config.se['video_name'], self.feed_path, (self.img_sz[1], self.img_sz[0]))
        else:
            list = glob.glob1(input_path + "images/", config.se['img_type'])
            list.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))
            self.feed_size = len(list)
            for i, filename in enumerate(list):
                shutil.copy(input_path + "images/" + filename, self.feed_path)
                os.rename(self.feed_path + '/' + filename, self.feed_path + '/frame' + str(i) + '.jpg')

        # prepare cropped images from all data set
        self.imgs = []
        self.imgs_trans = []
        self.trans = []
        self.imgs_big_embd = []
        self.imgs_relevant = []

        # Read images to array:
        for i in range(self.feed_size):
            img_str = self.feed_path + "/frame" + str(i) + ".jpg"
            I = cv2.imread(img_str)
            I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
            if self.img_sz[0] != -1:
                I = cv2.resize(I.copy(), (self.img_sz[1], self.img_sz[0]))
            self.imgs.append(I)

        print('img sz: ', self.img_sz)
        print('Finished uploading data, Number of video frames:', self.feed_size)


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass
