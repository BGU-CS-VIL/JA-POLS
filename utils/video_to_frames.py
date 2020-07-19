import cv2

# img_sz: (420, 250)
def extractImages(pathIn, pathOut, img_sz):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success, image = vidcap.read()
      if success:
          image = cv2.resize(image, img_sz)
          cv2.imwrite(pathOut + "/frame%d.jpg" % (count), image)     # save frame as JPEG file
          count += 1
    if count > 0:
        print('%d images where extracted successfully' % count)
    else:
        print('Images extraction failed.')
    return count

