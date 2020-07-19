
import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('7.jpg')
#
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
#
# kp = sift.detect(gray,None)
#
# img = cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)


def get_FC_by_SIFT(img1, img2):

    MIN_MATCH_COUNT = 10

    # img1 = cv2.imread(img1_str) # queryImage
    # img2 = cv2.imread(img2_str) # trainImage

    # Initiate SIFT detector
    # sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    img1_ = img1 * 255.
    img1__ = img1_.astype(np.uint8)
    img2_ = img2 * 255.
    img2__ = img2_.astype(np.uint8)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1__, None)
    kp2, des2 = sift.detectAndCompute(img2__, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        #good.append(m)
        if m.distance < 0.7*n.distance:
            good.append(m)

    # print(len(good))
    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()

        # Apply homography
        # height, width, channels = img1__.shape
        # result = cv2.warpPerspective(img1__, M, (width, height))

        # h,w,_ = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts, M)
        #
        # #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        #
        # w = np.ones((len(src_pts), 1))
        # src_pts_arr = np.reshape(src_pts, (-1,2))
        # dst_pts_arr = np.reshape(dst_pts, (-1,2))

        return M

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    # plt.imshow(img3, 'gray')
    # plt.show()
        return



