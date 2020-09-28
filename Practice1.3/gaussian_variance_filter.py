import cv2
import numpy as np

original = cv2.imread('media/cat.jpeg')

# cv2.imshow("original image", original)
# cv2.waitKey(0)

# create kernel
kk = np.ones((5, 5), np.float32)/25

blurred = cv2.filter2D(original, -1, kk)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)