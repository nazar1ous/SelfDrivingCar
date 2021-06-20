import cv2 as cv
import numpy as np
import math


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
    """
    Removes lines which are close to roi
    :param src: image
    :param lines: list of lines
    :return: ist of lines
    """
    size = src.shape[0], src.shape[1], 3
    h, w = size[:2]

    ppt = np.array([
        [0, 0.66 * h, 0.25 * w, 0.5 * h],
        [0.25 * w, 0.5 * h, 0.75 * w, 0.5 * h],
        [0.75 * w, 0.5 * h, w, 0.66 * h],
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

    return res


def equal_criteria(line1, line2):
    """
    Checks if two lines are similar
    :param line1: list of 4 coordinates
    :param line2: list of 4 coordinates
    :return: bool
    """
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    # Angle between lines should be quite small
    length1 = math.sqrt(pow(y11 - y12, 2) + pow(x11 - x12, 2))
    length2 = math.sqrt(pow(y21 - y22, 2) + pow(x21 - x22, 2))

    product = (x11 - x12) * (x21 - x22) + (y11 - y12) * (y21 - y22)

    if abs(product / (length1 * length2)) < math.cos(math.pi / 30):
        return False

    # Distance between centers of segments should be less than half of maximum length of two segments
    mx1, my1 = (x11 + x12) * 0.5, (y11 + y12) * 0.5
    mx2, my2 = (x21 + x22) * 0.5, (y21 + y22) * 0.5
    dist = math.sqrt((mx1 - mx2) * (mx1 - mx2) + (my1 - my2) * (my1 - my2))

    if dist > max(length1, length2) * 0.5:
        return False

    return True


def merge_lines(lines):
    """
    Merges lines by specific criteria
    :param lines: list of lines after HoughLinesP
    :return: list of merged lines
    """
    clusters = []
    checked_lines = set()

    for i, line in enumerate(lines):
        if np.array_str(line) in checked_lines:
            continue
        cluster = [line]
        checked_lines.add(np.array_str(line))
        for j, d_line in enumerate(lines[i + 1:]):
            if np.array_str(d_line) in checked_lines:
                continue
            if equal_criteria(line[0], d_line[0]):
                cluster.append(d_line)
                checked_lines.add(np.array_str(d_line))

        clusters.append(np.array(cluster))

    merged_lines = [np.mean(cluster, axis=0) for cluster in clusters]
    casted_lines = [i.astype(np.int32) for i in merged_lines]

    return casted_lines
