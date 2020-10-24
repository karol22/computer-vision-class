# Code from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

import cv2
import numpy as np
import math

vid = cv2.VideoCapture(0)

if(vid.isOpened() == False):
    print('Error opening video stream or file')

while(vid.isOpened()):
    ret, src = vid.read()
    if ret == True:
        dst = cv2.Canny(src, 50, 200, None, 3)
        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow("output", np.hstack([src, cdst]))

        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
vid.release()