# Authors: Sajarin Dider and Daniel Mallia

# Use of EAST detector inspired by: 
# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/

# NOTE: This script uses the TensorFlow Implementation of the EAST text 
# detector, which is offered by OpenCV. 
# The actual Github for this implementation is here: 
# https://github.com/argman/EAST
# The download link for the .pb file (trained model) is here: 
# https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

import cv as cv
import numpy as np 
import copy

def findCorners(bound):
    c1 = [bound[3][0],bound[0][1]]
    c2 = [bound[1][0],bound[0][1]]
    c3 = [bound[1][0],bound[2][1]]
    c4 = [bound[3][0],bound[2][1]]
    return [c1,c2,c3,c4]

def returnWordBBoxes(image):
    output = copy.deepcopy(image)

    # Need to scale image to a multiple of 32 - just steps down to next multiple of 32 
    newRows, newCols = output.shape[:2]
    newRows -= newRows % 32
    newCols -= newCols % 32
    output = cv.resize(image, (newRows , newCols), interpolation = cv.INTER_LINEAR)

    # Import pretrained EAST detector
    east = cv.dnn.readNet('frozen_east_text_detection.pb')

    # Required layer names
    layerNames = [ "feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    

if __name__ == "__main__":

    bndingBx = []#holds bounding box of each countour
    corners = []
    image = cv.imread('e.png',0)
    blur = cv.GaussianBlur(image,(5,5),0)
    threshold, threshImage = cv.threshold(blur,0,255,
        cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
     
    contours, heirar = cv.findContours(threshImage, cv.RETR_CCOMP, 
        cv.CHAIN_APPROX_SIMPLE)
    for num in range(0,len(contours)):
        #make sure contour is for letter and not cavity
        if(heirar[0][num][3] == -1):
            left = tuple(contours[num][contours[num][:,:,0].argmin()][0])
            right = tuple(contours[num][contours[num][:,:,0].argmax()][0])
            top = tuple(contours[num][contours[num][:,:,1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:,:,1].argmax()][0])
            bndingBx.append([top,right,bottom,left])
    
        for bx in bndingBx:
            corners.append(findCorners(bx))
    
        #draw the countours on threshImage image
        x,y,w,h = cv.boundingRect(threshImage)

    cv.imshow('thresh', threshImage)
    cv.waitKey()


