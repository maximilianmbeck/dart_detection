import cv2 as cv
from matplotlib import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
from numpy.core.fromnumeric import shape

def main():
    # read image
    image = cv.imread('input_multicolor_thresh.jpg', cv.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots()
    ax.imshow(image)

    # morphologic operations
    kernel_size = 5
    kernel = np.ones(shape=(kernel_size,kernel_size), dtype=np.uint8)
    # open
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    # fig, ax = plt.subplots()
    # ax.imshow(opening)
    # close
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    fig, ax = plt.subplots()
    ax.imshow(closing)


    # blob detection (BLOB = Binary Large Object)
    params = cv.SimpleBlobDetector_Params()
    # Change thresholds -> we actually already have a binary image -> thresholded by colors
    params.minThreshold = 200
    params.maxThreshold = 220
    # Filter by Area -> take only blobs with area greater than 100 pixels
    params.filterByArea = True
    params.minArea = 100

    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(closing)

    fig, ax = plt.subplots()
    out = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(out)

    circles = []
    for kp in keypoints:
        print(kp.pt, '+-', kp.size/2, ' pixels')
        ax.plot(kp.pt[0], kp.pt[1], marker='x', markersize=5, color='g')
        ax.add_patch(mpatches.Circle((kp.pt[0], kp.pt[1]), kp.size/2, color='y', fill=False))
    
    plt.show()


if __name__ == '__main__':
    main()