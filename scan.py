#! /usr/bin/env python3

# Authors: Daniel Mallia and Sajarin Dider
# Date Begun: 1/17/2020

import sys
import cv2 as cv
import numpy as np
import copy
import math
from keras.models import load_model
import char 

classLabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 
 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
 'w', 'x', 'y', 'z']

# Helper function - prepares image for input to network
def processImage(image, dimension):
    output = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    output = cv.resize(output, (dimension, dimension), interpolation = cv.INTER_LINEAR) # Resize - may want to try cv.INTER_AREA
    output = output.astype('float32') / 255 # Scale (normalize) to match network training
    output = np.expand_dims(output, axis=0) # Reshape as a 4D tensor for input to network
    
    return output 

# Helper function - facilitates sorting by contour area
def sortByArea(contour):
    return cv.contourArea(contour)

# Helper function - calculates distance
def euclideanDistance(point1, point2):
    return math.sqrt(((point1[0] - point2[0])**2) + \
        ((point1[1] - point2[1])**2))

# Transformation function - "snaps" document to image frame
def transformForOCR(image):
    rows, cols, channels = image.shape
    max_row = rows - 1
    max_col = cols - 1

    # Create local copies, convert detection copy to grayscale 
    document = copy.deepcopy(image)
    document = cv.cvtColor(document, cv.COLOR_RGB2GRAY)
    contoured = copy.deepcopy(image)
    output = copy.deepcopy(image)

    # Threshold image - document should be significantly lighter than 
    # background
    value, document = cv.threshold(document, 180, 255, cv.THRESH_BINARY \
        + cv.THRESH_OTSU)

    # Find contours and select contour which covers the most area - 
    # should be the document; draw contour on contoured
    contours, hierarchy = cv.findContours(document, cv.RETR_TREE, \
        cv.CHAIN_APPROX_NONE)
    contours.sort(key=sortByArea, reverse=True)
    documentContour = contours[0]
    cv.drawContours(contoured, contours, 0, (0, 0, 255), thickness=100)

    # Approximate contour with rectangle and draw box points on contoured
    tolerance = 0.1*cv.arcLength(documentContour,True)
    approximateRectangle = cv.approxPolyDP(documentContour,tolerance,True)
    approxList = []
    for point in approximateRectangle: # Convert to normal list
        approxList.append(tuple(point[0]))
    for point in approxList:
        cv.circle(contoured, point, radius=5, color=(0,255,0), thickness=100)
    
    # "Destination" points are the corners of the frame
    topLeftDest = [0, 0]
    topRightDest = [0, max_col]
    bottomLeftDest = [max_row, 0]
    bottomRightDest = [max_row, max_col]
    dest = np.float32([topLeftDest,bottomLeftDest, topRightDest, \
        bottomRightDest])

    # Calculate which "source" points are top left/right, bottom left/right
    topLeft = min(approxList, key=lambda x : euclideanDistance(x, topLeftDest))
    topRight = min(approxList, key= lambda x : euclideanDistance(x, topRightDest))
    bottomLeft = min(approxList, key= lambda x : euclideanDistance(x, bottomLeftDest))
    bottomRight = min(approxList, key= lambda x : euclideanDistance(x, bottomRightDest))
    src = np.float32([topLeft, bottomLeft, topRight, bottomRight])

    # Get perspective transform matrix and apply transform
    matrix = cv.getPerspectiveTransform(src, dest)
    output = cv.warpPerspective(output, matrix, (max_row, max_col))

    return document, contoured, output

# Main
def main():
    # Usage Handling:
    if(len(sys.argv) != 2):
        print('Usage: ./scan.py [IMAGE.jpg]')
        sys.exit()

    # Load Model
    network = load_model('OCRNetworkVersion2.h5')

    # Read in image
    imageName = sys.argv[1]
    image = cv.imread(imageName, cv.IMREAD_COLOR)

    # Detect and transform document prior to OCR
    document, contoured, transformed = transformForOCR(image)

    # Detect individual characters in document
    lines = char.returnLines(transformed)
    characterList = char.returnCharacterImageList(lines, transformed)

    # Collect outputs of the neural network
    text = ""
    i = 0
    for line in characterList: # For every line
        for word in line: # For every word
            for characterImage in word: # For every character in that word
                inputImage = processImage(characterImage, 32) # Prepare as input
                view = cv.cvtColor(characterImage, cv.COLOR_BGR2RGB)
                view = cv.resize(view, (32, 32), interpolation = cv.INTER_LINEAR)
                cv.imwrite('./Images/Char' + str(i) + '.png', view)
                results = network.predict(inputImage) # Pass through network
                result = classLabels[np.argmax(results)]
                text += result # Append result
                i+=1
            text += " " # Add space after every word
        text += "\n" # Add newline after every line

    # # Display transform results
    # cv.namedWindow('Transformed', cv.WINDOW_NORMAL)
    # cv.resizeWindow('Transformed', 800, 600)
    # cv.imshow('Transformed', transformed)
    # if(cv.waitKey(0) == ord('q')):
    #     cv.destroyWindow('Transformed')
    #     cv.waitKey(1)
    #     sys.exit()
    # cv.destroyWindow('Transformed')
    # cv.waitKey(1)

    # Save the document image to a new file
    # imageName = imageName.rstrip(".jpg")
    # cv.imwrite(imageName + 'Transformed.jpg', transformed)

    # Write text to file 
    outputFileName = imageName.rstrip(".jpg") + "Text.txt"
    try:
        outputTextFile = open(outputFileName, 'x')
    except:
        print("Output text file already exists.")
        sys.exit()

    outputTextFile.write(text)
    outputTextFile.close()

if __name__ == "__main__":
    main()
