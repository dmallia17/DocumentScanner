#! /usr/bin/env python3

# Author: Daniel Mallia
# Date Begun: 1/17/2020

import sys
import cv2 as cv


# Transformation function - "snaps" document to image frame
def transformForOCR(image):
    pass

# Main
def main():
    # Usage Handling:
    if(len(sys.argv) != 2):
        print('Usage: ./scan.py [IMAGE]')
        sys.exit()

    # Read in image
    imageName = sys.argv[1]
    image = cv.imread(imageName, cv.IMREAD_COLOR)

    # Detect and transform document prior to OCR
    #document = transformForOCR(image)

    # Display transform results
    cv.namedWindow('Transformed', cv.WINDOW_NORMAL)
    cv.resizeWindow('Transformed', 800, 600)
    cv.imshow('Transformed', image)
    if(cv.waitKey(0) == ord('q')):
        cv.destroyWindow('Transformed')
        cv.waitKey(1)
        sys.exit()
    cv.destroyWindow('Transformed')
    cv.waitKey(1)

    # Save the document image to a new file

    # Write text to file 

if __name__ == "__main__":
    main()