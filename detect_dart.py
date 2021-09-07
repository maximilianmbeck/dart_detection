import cv2 as cv
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path', help='Path to the folder, where the images are located.')


# Set minimum and maximum HSV values to display
#! > use hsv_trackbar_template to find values
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

# assert image filenames have following layout:
# filename=f"magictest-{datetime.datetime.now():%Y%m%d-%H%M%S}-{series_id}-{throw_id}-camera-{anynumber}.JPG"
# magictest-20210904-163555-1-0-camera-00007.JPG
split_token = '-'
series_id_idx = 3
throw_id_idx = 4

"""
This file contains the code to detect darts from the taken images.

USAGE:
python3 detect_dart.py --path /path/to/folder
e.g.    python3 detect_dart.py --path /home/max/phd/radar/dart_detection/test_shots
if --path is not specified the path of this file will be used
"""


class DartDetector(object):

    def __init__(self, img_filename_suffix_startswith='-camera-'):
        self.img_filename_suffix_startswith = img_filename_suffix_startswith

    def main(self, path: str):
        self.path = path
        errlog = ""
        jpg_image_filenames = get_image_filenames_in_directory(path)

        self.img_fnames_by_ids = get_series_by_ids(jpg_image_filenames)
        for series_id in sorted(self.img_fnames_by_ids.keys()):
            detection_pool = []  # keeps only new positions
            sorted_results = []  # logs all detections
            series_lost = False
            for throw_id in sorted(self.img_fnames_by_ids[series_id].keys()):

                image_name = self.img_fnames_by_ids[series_id][throw_id]
                if series_lost:
                    detection_pool.append(np.array([]))
                    continue

                image_file = path+'/'+image_name
                detections_in_img, _ = self.detect_darts(
                    green_bound_tuple, image_file)
                sorted_results.append(detections_in_img)

                if throw_id > 0:  # < index 0 is empty board
                    if detections_in_img.size == 0:
                        # error aruco marker not detected
                        errlog += 'SHOT LOST: Aruco marker not detected in series_id: {0}, throw_id: {1} \n'.format(
                                series_id, throw_id)
                        sorted_results.append(np.array([]))
                        continue

                    new_detections = get_new_detections(
                        detection_pool, detections_in_img)
                    # check for errors
                    if len(new_detections) == 1:
                        # no error
                        detection_pool.append(new_detections[0])
                    elif len(new_detections) == 0:
                        # no new dart -> dart not detected -> probably hidden by other dart
                        # -> only this detection is lost
                        detection_pool.append(np.array([]))
                        errlog += 'SHOT LOST: No new dart detected in series_id: {0}, throw_id: {1} \n'.format(
                            series_id, throw_id)
                        print('SHOT LOST: No new dart detected in series_id: {0}, throw_id: {1}'.format(
                            series_id, throw_id))

                    elif len(new_detections) > 1:
                        # lost one image -> throw the following darts away
                        series_lost = True
                        detection_pool.append(np.array([]))
                        errlog += 'SERIES LOST: More than one new dart detected in series_id: {0}, throw_id: {1} \n'.format(
                            series_id, throw_id)
                        print('SERIES LOST: More than one new dart detected in series_id: {0}, throw_id: {1}'.format(
                            series_id, throw_id))

            self.save_detections(path, series_id, detection_pool)

            for s in detection_pool:
                print(s)

        outstr = ''
        outstr+= "===================\n"
        outstr+= "ERRORLOG:\n"
        if errlog == "":
            outstr+= "no errors!\n"
        else:
            outstr+= errlog
        outstr+= "===================\n"

        with open(path+'/dart_detection_errorlog.txt', 'w') as outfile:
            outfile.write(outstr)
        
        print(outstr)

    # computer vision pipeline

    def detect_darts(self, color_bounds: tuple, filename: str):
        print(filename)

        # read image
        image = cv.imread(filename)
        # find markers
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        # frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids) # only for visualizing

        # get sorted corners
        clw_sorted_corners = get_sorted_corner_means(ids, corners)

        # define ground truth marker points
        zoom_ = 4
        dest_corners = zoom_*np.array([[0, 0], [500, 0], [1000, 0],
                                      [0, 350], [1000, 350], [0, 700], [500, 700], [1000, 700]])

        dest_width = zoom_*1000
        dest_height = zoom_*700

        if len(clw_sorted_corners) != len(dest_corners):
            # ERROR
            print("Not all aruco markers detected!")
            # not all aruco markers detected
            return np.array([]), []

        # find homography
        h, mask = cv.findHomography(
            clw_sorted_corners, dest_corners, cv.RANSAC)
        # 'rectify' image
        warp_img = cv.warpPerspective(image, h, (dest_width, dest_height))

        # detect darts by thresholding in HSV colorspace
        lower, upper = color_bounds
        # Convert to HSV format and color threshold
        hsv = cv.cvtColor(warp_img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(warp_img, warp_img, mask=mask)
        result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
        _, dst = cv.threshold(result_gray, 10, 255, 0)

        image_thresh = dst

        # morphologic operations
        kernel_size = 5
        kernel = np.ones(shape=(kernel_size, kernel_size), dtype=np.uint8)
        # open
        opening = cv.morphologyEx(image_thresh, cv.MORPH_OPEN, kernel)
        # close
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

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
        # # out = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.save_detections_img(self.path, filename, warp_img, keypoints)

        # extract position from keypoints in pixel values
        positions_m = extract_pos_from_keypoints(keypoints, zoom_)
        return positions_m, keypoints

    def save_detections(self, path: str, series_id: int, detection_pool: list):
        """Saves all detections in separate yaml files."""
        for throw_id in range(len(detection_pool)):
            try:
                p = Path(path)/Path(self.img_fnames_by_ids[series_id][throw_id+1])
            except KeyError:
                print('FILE ERROR: Input file for series id: {0}, throw id {1} is missing!'.format(series_id, throw_id+1))
                continue
            img_filename_wo_ext = p.name
            # remove all tokens after and including img_filename_suffix_startswith
            pure_filename = img_filename_wo_ext[0:img_filename_wo_ext.index(
                self.img_filename_suffix_startswith)]

            yml_filename = pure_filename+'-position.yaml'

            detection = detection_pool[throw_id]
            if detection.size == 0:
                detection_valid = False
                detection = np.array(
                    [float('nan'), float('nan'), float('nan')])
            else:
                detection_valid = True

            detection_dict = dict(x=float(detection[0]), y=float(detection[1]), radius=float(
                detection[2]), detection_valid=int(detection_valid), series_id=series_id, throw_id=throw_id+1)

            with open(path+'/'+yml_filename, 'w') as outfile:
                yaml.dump(detection_dict, outfile, default_flow_style=False)

    def save_detections_img(self, path: str, filename_input_img: str, warped_img: np.ndarray, detected_keypoints: list):
        fig, ax = plt.subplots()
        # out = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # ax.imshow(out)
        ax.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
        for kp in detected_keypoints:
            # print(kp.pt, '+-', kp.size/2, ' pixels')
            ax.plot(kp.pt[0], kp.pt[1], marker='x', markersize=5, color='g')
            ax.add_patch(mpatches.Circle(
                (kp.pt[0], kp.pt[1]), kp.size/2, color='y', fill=False))

        p = Path(path)/Path(filename_input_img)
        img_filename_wo_ext = str(p)
        # remove all tokens after and including img_filename_suffix_startswith
        pure_filename = img_filename_wo_ext[0:img_filename_wo_ext.index(
            self.img_filename_suffix_startswith)]

        rect_img_filename = pure_filename+'-camerarect.jpg'

        fig.savefig(rect_img_filename, dpi=300)
        plt.close()

# detection postprocessing


def get_new_detections(detection_pool: list, incoming_detections: np.ndarray):
    """
    Arrays of shape nx3 (rows: different detections, columns: x,y, radius)
    """
    new_detections = []
    for d in incoming_detections:
        if is_detection_new(detection_pool, d):
            new_detections.append(d)

    return new_detections


def is_detection_new(detection_pool: list, query_pos: np.ndarray) -> bool:
    for p in detection_pool:
        if is_same_pos(p, query_pos):
            return False
    return True


def is_same_pos(pos: np.ndarray, query_pos: np.ndarray) -> bool:
    if pos.size != query_pos.size:
        return False
    else:
        dist_vec = pos[0:2] - query_pos[0:2]
        dist = np.linalg.norm(dist_vec)
        return dist < pos[2]

# directory operations


def get_image_filenames_in_directory(path: str, name_contains='-camera-'):
    """Returns a list of all image files in the directory."""
    p = Path(path)
    jpg_images = [x.name for x in p.iterdir() if x.suffix ==
                  '.JPG' or x.suffix == '.jpg']
    if name_contains is not None:
        filenames = [x for x in jpg_images if name_contains in x]
    else:
        filenames = jpg_images
    return filenames


def get_series_by_ids(image_filenames: list, split_token='-', series_id_idx=3, throw_id_idx=4) -> dict:
    """
    Returns a dict (series_ids as keys), which contains another dict (throw_ids as keys).
    """
    series_by_ids = {}

    for img_fn in image_filenames:
        tokenized_img_fn = img_fn.split(split_token)
        series_id = int(tokenized_img_fn[series_id_idx])
        throw_id = int(tokenized_img_fn[throw_id_idx])
        if series_id not in series_by_ids:
            series_by_ids[series_id] = {}
        series_by_ids[series_id][throw_id] = img_fn

    return series_by_ids

# helpers for computer vision pipeline


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
    args = parser.parse_args()
    if args.path:
        # path = Path().resolve()/Path(args.path) #< Path().resolve() returns a Path to the current working directory
        path = Path(args.path)
    else:
        path = Path(__file__).resolve()

    dd = DartDetector()
    dd.main(str(path))
