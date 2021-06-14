import cv2 as cv
import numpy as np


# def roi(src):
#     # initialize image mask
#     size = src.shape[0], src.shape[1], 3
#     mask = np.zeros(size, dtype=np.uint8)
#
#     # creating polygon shape
#     h, w = size[:2]
#     ppt = np.array([[0, h], [0, 0.66 * h], [0.25 * w, 0.33 * h], [0.75 * w, 0.33 * h], [w, 0.66 * h], [w, h]],
#                    np.int32)
#     ppt = ppt.reshape((-1, 1, 2))
#
#     # creating mask with filled polygon
#     cv.fillPoly(mask, [ppt], (255, 255, 255))
#
#     # apply masking
#     masked = cv.bitwise_and(mask, src)
#
#     return masked


img = cv.imread(cv.samples.findFile("c2.jpg"))

img = roi(img)


cv.imshow("masked", img)
# cv.moveWindow(rook_window, W, 200)


cv.waitKey(0)
cv.destroyAllWindows()
