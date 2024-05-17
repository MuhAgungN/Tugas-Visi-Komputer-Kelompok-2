import cv2 as cv
import numpy as np

capture = cv.VideoCapture('data/mouse.mp4')
#capture = cv.VideoCapture(0)
if not capture.isOpened():
    exit(0)

subsKNN = cv.createBackgroundSubtractorKNN()

while capture.isOpened():
    re, frame = capture.read()

    if isinstance(frame, type(None)):
        break

    blobKNN = subsKNN.apply(frame)

    cv.imshow("image knn", blobKNN)
    keyword = cv.waitKey(30)
    if keyword == 'q' or keyword == 27:
        break

cv.destroyAllWindows()
exit(0)
