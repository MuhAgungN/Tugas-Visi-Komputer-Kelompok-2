import cv2 as cv
import numpy as np

# capture = cv.VideoCapture('data/mouse.mp4')
capture = cv.VideoCapture(0)
if not capture.isOpened():
    exit(0)
# subsMog2 = cv.createBackgroundSubtractorMOG2(600,125,False)
subsMog2 = cv.createBackgroundSubtractorMOG2(300, 400, True)
i = 0
while capture.isOpened():
    re, frame = capture.read()
    scale = 70

    if isinstance(frame, type(None)):
        break

    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)

    dim = (width, height)
    image = cv.resize(frame,dim, cv.INTER_AREA)
    
    gaussian = np.array([
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0]
    ])/273
    
    # gaussian = np.array([
    #     [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
    #     [0.0, 3.0, 13.0, 22.0, 13.0, 3.0, 0.0],
    #     [1.0, 13.0, 59.0, 97.0, 59.0, 13.0, 1.0],
    #     [2.0, 22.0, 97.0, 159.0, 97.0, 22.0, 2.0],
    #     [1.0, 13.0, 59.0, 97.0, 59.0, 13.0, 1.0],
    #     [0.0, 3.0, 13.0, 22.0, 13.0, 3.0, 0.0],
    #     [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
    # ])/273
    
    image = cv.filter2D(image,-1,gaussian)

    blobMog = subsMog2.apply(image)

    cv.imshow("image asli",image)
    cv.imshow("image mog",blobMog)
    keyword = cv.waitKey(30)
    if keyword=='q' or keyword==27:
        break
cv.destroyAllWindows()
exit(0)