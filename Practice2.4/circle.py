# https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
import cv2
import numpy as np

vid = cv2.VideoCapture(0)

if(vid.isOpened() == False):
    print('Error opening video stream or file')

while(vid.isOpened()):
    ret, image = vid.read()
    if ret == True:
        image = cv2.resize(image, (960, 540))
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv2.imshow("output", np.hstack([image, output]))

        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
vid.release()