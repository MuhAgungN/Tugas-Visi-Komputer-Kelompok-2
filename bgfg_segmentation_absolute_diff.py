import cv2 as cv
import numpy as np

def denoise(frame):
    frame = cv.medianBlur(frame, 11)
    frame = cv.GaussianBlur(frame, (5, 5),0)
    
    return frame

# capture = cv.VideoCapture('data/mouse.mp4')
capture = cv.VideoCapture(0)

if not capture.isOpened():
    exit(0)

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
    
    #kernel gaussian blur
    gaussian = np.array([
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0]
    ])/273
    
    #apply kernel
    # image = cv.filter2D(image,-1,gaussian)
    image = denoise(image)
    
    
    if i == 0:
        frame1 = image
        grayscaleframe1 = cv.cvtColor(frame1, cv.COLOR_RGBA2GRAY)
        i = i+1
        
    grayscaleframe = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    
    framedelta = cv.absdiff(grayscaleframe, grayscaleframe1)
    
    _, bgs = cv.threshold(framedelta, 45, 255, cv.THRESH_BINARY_INV)
    
    # show result
    cv.imshow("image asli",image)
    cv.imshow("image abs diff", bgs)
    
    keyword = cv.waitKey(30)
    if keyword=='q' or keyword==27:
        break
    
cv.destroyAllWindows()
exit(0)