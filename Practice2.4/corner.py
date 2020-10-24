# Code from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
import cv2
import numpy as np


source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255
def cornerHarris_demo(val):
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)



vid = cv2.VideoCapture(0)

if(vid.isOpened() == False):
    print('Error opening video stream or file')

while(vid.isOpened()):
    ret, img = vid.read()
    if ret == True:
        img = cv2.resize(img, (960, 540))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.02*dst.max()]=[0,255,0]

        cv2.namedWindow("image", cv2.WINDOW_NORMAL);
        cv2.imshow('image', img)
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
vid.release()