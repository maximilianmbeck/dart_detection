import cv2 as cv
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set minimum and maximum HSV values to display
# lower = np.array([hMin, sMin, vMin])
# upper = np.array([hMax, sMax, vMax])
# for initial image
lower_g = np.array([60, 100, 40])
upper_g = np.array([80, 255, 100])
lower_r = np.array([150, 130, 30])
upper_r = np.array([179, 255, 90])
# tuned with testthrows
lower_g = np.array([40, 100, 10])
upper_g = np.array([80, 255, 255])
lower_r = np.array([40, 100, 10])
upper_r = np.array([80, 255, 255])

green_bound_tuple = (lower_g, upper_g)
red_bound_tuple = (lower_r, upper_r)

def main():
    detect_dart(green_bound_tuple)

def detect_dart(color_bounds):
    # read image
    filename = 'sunday'
    image = cv.imread(filename+'.jpg')
    plt.figure()
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()

    # find markers
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

    # show markers
    plt.figure()
    plt.imshow(frame_markers)
    for i in range(len(ids)):
        c = corners[i][0]
        # plot upper left corner
        # plt.plot([c[0, 0]], [c[0, 1]],
        #      "o", label="id={0}".format(ids[i]))
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()],
                 "o", label="id={0}".format(ids[i]))
    plt.legend()
    plt.show()

    # get sorted corners
    clw_sorted_corners = get_sorted_corner_means(ids, corners)

    # define ground truth marker points
    zoom_ = 4
    dest_corners = zoom_*np.array([[0, 0], [500, 0], [1000, 0],
                                  [0, 350], [1000, 350], [0, 700], [500, 700], [1000, 700]])
    ##! without aruco marker 7 (on picture)
    # dest_corners = zoom_*np.array([[0, 0], [500, 0], [1000, 0],
    #                               [1000, 350], [0, 700], [500, 700], [1000, 700]])
    dest_width = zoom_*1000
    dest_height = zoom_*700

    # find homography
    h, mask = cv.findHomography(clw_sorted_corners, dest_corners, cv.RANSAC)
    # 'rectify' image
    warp_img = cv.warpPerspective(image, h, (dest_width, dest_height))
    cv.imwrite(filename+'_warped.jpg', warp_img)
    plt.figure()
    plt.imshow(cv.cvtColor(warp_img, cv.COLOR_BGR2RGB))
    plt.show()

    # detect darts
    lower, upper = color_bounds
    # Convert to HSV format and color threshold
    hsv = cv.cvtColor(warp_img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(warp_img, warp_img, mask=mask)
    result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    _, dst = cv.threshold(result_gray, 10, 255, 0)
    cv.imwrite(filename+'_thresh.jpg', dst)
    plt.imshow(dst)
    plt.show()
    
    # # find dart position
    # dart_positions = np.argwhere(mask>200)
    # dart_pos_pixel = np.array([dart_positions[:,0].mean(), dart_positions[:,1].mean()])
    
    # # backtransform to mm range divide by zoom_
    # dart_pos_mm = dart_pos_pixel/zoom_

    # return dart_pos_mm
    image = dst

    # fig, ax = plt.subplots()
    # ax.imshow(image)

    # morphologic operations
    kernel_size = 5
    kernel = np.ones(shape=(kernel_size,kernel_size), dtype=np.uint8)
    # open
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    # fig, ax = plt.subplots()
    # ax.imshow(opening)
    # close
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # fig, ax = plt.subplots()
    # ax.imshow(closing)


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
    # ax.imshow(out)
    ax.imshow(cv.cvtColor(warp_img, cv.COLOR_BGR2RGB))

    circles = []
    for kp in keypoints:
        print(kp.pt, '+-', kp.size/2, ' pixels')
        ax.plot(kp.pt[0], kp.pt[1], marker='x', markersize=5, color='g')
        ax.add_patch(mpatches.Circle((kp.pt[0], kp.pt[1]), kp.size/2, color='y', fill=False))
    
            # extract position from keypoints in pixel values
    positions_m = extract_pos_from_keypoints(keypoints, zoom_)
    print(positions_m)
    plt.show()

def extract_pos_from_keypoints(keypoints, zoom, origin=np.array([500, 350])) -> np.ndarray:
    """
    Returns a list of tuples, containing (position in m, radius of detected tip in m).
    origin in mm
    """
    positions_m = []
    for kp in keypoints:
        # position in mm with origin upper-left corner aruco marker, positive y going down
        # format [x,y,radius]
        pos_radius_mm = np.array(
            [kp.pt[0]/zoom, kp.pt[1]/zoom, (kp.size/2)/zoom])
        print(pos_radius_mm)
        # transform to true origin
        pos_origin_mm = pos_radius_mm[0:2] - origin
        pos_origin_mm[1] *= -1
        # in m
        pos_origin_m = pos_origin_mm/1e3
        radius_m = pos_radius_mm[2]/1e3

        positions_m.append(np.hstack((pos_origin_m, radius_m)))
    return np.array(positions_m)

def get_mean_corners(corners):
    corner_means = []
    for i in range(len(corners)):
        c = corners[i][0]
        cm = [c[:, 0].mean()], [c[:, 1].mean()]
        corner_means.append(cm)
    return np.array(corner_means).squeeze()

def get_upper_left_corner(corners):
    corner_ul = []
    for i in range(len(corners)):
        c = corners[i][0]
        cm = [c[0, 0]], [c[0, 1]]
        corner_ul.append(cm)
    return np.array(corner_ul).squeeze()


def get_sorted_corner_means(ids, corners, get_point=get_mean_corners):
    get_mean_corners(corners), ids
    concatar = np.hstack((ids, get_point(corners)))
    sortedarr = concatar[concatar[:, 0].argsort()]
    return sortedarr[:, 1:3]


if __name__ == '__main__':
    main()
