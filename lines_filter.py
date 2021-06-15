import cv2 as cv
import numpy as np


def filter_lines(lines):
    """
    Splits lines on two groups (left lines and right)
    After takes mean of coordinates and return twi lines
    :param lines: list of lines
    :return: two lines
    """
    # slope_l, slope_r = [], []
    lane_l, lane_r = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.1:
                # slope_r.append(slope)
                lane_r.append(line)
            elif slope < -0.1:
                # slope_l.append(slope)
                lane_l.append(line)

    if (len(lane_l) == 0) or (len(lane_r) == 0):
        print('no lane detected')
        return 1

    mean_l = np.mean(np.array(lane_l), axis=0)
    mean_l = [np.array(mean_l, np.int32)]

    mean_r = np.mean(np.array(lane_r), axis=0)
    mean_r = [np.array(mean_r, np.int32)]

    return mean_l, mean_r


def distance(line, p):
    p1 = np.array([line[0], line[1]])
    p2 = np.array([line[2], line[3]])
    p3 = np.array([p[0], p[1]])
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def remove_roi_lines(src, lines):
    size = src.shape[0], src.shape[1], 3
    h, w = size[:2]

    ppt = np.array([
        [0, 0.66 * h, 0.25 * w, 0.4 * h],
        [0.25 * w, 0.4 * h, 0.75 * w, 0.4 * h],
        [0.75 * w, 0.4 * h, w, 0.66 * h],
    ], np.int32)

    res = []
    max_distance = 10

    for line in range(len(lines)):
        l = lines[line][0]
        same = [False, False]
        for roi_line in range(len(ppt)):
            same[0] = distance(ppt[roi_line], l[:2]) <= max_distance
            same[1] = distance(ppt[roi_line], l[2:4]) <= max_distance

            if not (False in same):
                break

        if False in same:
            res.append(lines[line])

    # print(len(res))
    return res
