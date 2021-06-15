import cv2 as cv
import numpy as np


def load_img(filename):
    """
    Loads image
    :param filename: mane of file with image
    :return:  image in np.ndarray
    """
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename))
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        return -1
    return src


def white_mask(src):
    """
    Creates and applies color mask
    :param src:  image in np.ndarray
    :return:  image in np.ndarray
    """
    img = cv.cvtColor(src, cv.COLOR_BGR2HLS)

    lower_white = np.array([0, 165, 0])
    upper_white = np.array([255, 255, 255])
    white = cv.inRange(img, lower_white, upper_white)

    return cv.bitwise_and(src, src, mask=white)


def roi(src):
    """
    Selects region of interest from image
    :param src: image in np.ndarray
    :return:  image in np.ndarray
    """
    # initialize image mask
    size = src.shape[0], src.shape[1], 3
    mask = np.zeros(size, dtype=np.uint8)

    # creating polygon shape
    h, w = size[:2]
    ppt = np.array([[0, h], [0, 0.66 * h], [0.25 * w, 0.4 * h], [0.75 * w, 0.4 * h], [w, 0.66 * h], [w, h]],
                   np.int32)
    ppt = ppt.reshape((-1, 1, 2))

    # creating mask with filled polygon
    cv.fillPoly(mask, [ppt], (255, 255, 255))

    # apply masking
    masked = cv.bitwise_and(mask, src)

    return masked


def remove_roi_lines(src, lines):
    size = src.shape[0], src.shape[1], 3
    h, w = size[:2]

    ppt = np.array([
        [0, 0.66 * h, 0.25 * w, 0.4 * h],
        [0.25 * w, 0.4 * h, 0.75 * w, 0.4 * h],
        [0.75 * w, 0.4 * h, w, 0.66 * h],
    ], np.int32)

    res = []
    radius = 100

    for line in range(len(lines)):
        l = lines[line][0]
        same = [False, False, False, False]
        for roi_line in range(len(ppt)):
            for i in range(4):
                same[i] = (((ppt[roi_line][i] - radius) <= l[i]) and (l[i] <= (ppt[roi_line][i] + radius)))
            if not (False in same):
                break

        if False in same:
            res.append(lines[line])

    res2 = []
    slopes = [(i[1] - i[3]) / (i[0] - i[2]) for i in ppt]
    print(slopes)
    for line in res:
        l = line[0]
        slope = (l[1] - l[3]) / (l[0] - l[2])
        same = [False, False, False]
        for s in range(len(slopes)):
            same[s] = (((slopes[s] - 0.1) <= slope) and (slope <= (slopes[s] + 0.1)))

        if not (True in same):
            res2.append(line)

    return res2


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


def draw_lines(img, lines):
    """
    Draws lines on image
    :param img: image in np.ndarray
    :param lines: lines to draw
    :return: image with drawn lines
    """
    print(type(img))
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv.line(img, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)


def detect_lines(filename: str):
    """
    Detects lane in image
    :param filename: image file name
    :return: two lists of lines
    """
    img = load_img(filename)
    cv.imshow("img", img)

    # region of interest
    img = roi(img)
    # cv.imshow("roi", img)

    img = cv.GaussianBlur(img, (5, 5), 0)

    # color mask
    # img = white_mask(img)
    # cv.imshow("Mask", img)

    # Edges
    img = cv.Canny(img, 75, 150)
    # cv.imshow("Canny", img)

    # Some blur
    img = cv.GaussianBlur(img, (5, 5), 0)
    # cv.imshow("GaussianBlur", img)

    # Lines
    lines = cv.HoughLinesP(img, 1, np.pi / 180, 200, None, 150, 10)

    print(len(lines))
    # removes roi lines
    lines = remove_roi_lines(img, lines)
    print(len(lines))

    # filter for lines
    # lines = filter_lines(lines)

    img_cvt = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    draw_lines(img_cvt, lines)
    #
    # draw_lines(img_cvt, lines[0])
    # draw_lines(img_cvt, lines[1])

    cv.imshow("result", img_cvt)

    cv.waitKey()
    cv.destroyAllWindows()
    return lines


if __name__ == "__main__":
    detect_lines("bohdanData/c2.jpg")
    # detect_lines("data/32.png")
    # detect_lines("bohdanData/img2.png")
