import cv2 as cv
import numpy as np
from lines_filter import filter_lines, remove_roi_lines, merge_lines


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
    img = cv.cvtColor(src, cv.COLOR_RGB2HLS)

    lower_white = np.array([0, 0, 0])
    upper_white = np.array([255, 255, 255])
    white = cv.inRange(img, lower_white, upper_white)

    lower = np.uint8([240, 240, 240])
    upper = np.uint8([255, 255, 255])
    yellow = cv.inRange(img, lower, upper)

    mask = cv.bitwise_or(white, yellow)
    return cv.bitwise_and(src, src, mask=mask)


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
    ppt = np.array([[0, h], [0, 0.66 * h], [0.25 * w, 0.5 * h], [0.75 * w, 0.5 * h], [w, 0.66 * h], [w, h]],
                   np.int32)
    ppt = ppt.reshape((-1, 1, 2))

    # creating mask with filled polygon
    cv.fillPoly(mask, [ppt], (255, 255, 255))

    # apply masking
    masked = cv.bitwise_and(mask, src)

    return masked


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
    lines = cv.HoughLinesP(img, 1, np.pi / 180, 200, None, 150, 0)

    # removes roi lines
    lines = remove_roi_lines(img, lines)

    # filter for lines
    # lines = filter_lines(lines)

    # cluster lines
    print(f"len(lines): {len(lines)}")
    lines = merge_lines(lines)
    # lines = merge_lines(lines)
    print(f"len(lines): {len(lines)}")
    # print(clusters)

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
    detect_lines("bohdanData/img.png")
    # detect_lines("bohdanData/.jpg")
    # detect_lines("data/48.png")
    # detect_lines("bohdanData/img2.png")
