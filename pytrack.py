#!/usr/bin/env python3
# we are recreating the old matlab fly tracker in python

import cv2
import sys
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets

# configurable params
search_size = 20
per_pixel_threshold = 1.5
arena_size = 8  # assumes square arena


def main(argv):
    path = 'Aarya_videos/IMG_5807.m4v'

    # open video, grab basic metadata
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        sys.exit('Could not open: ' + path)
    framerate = video.get(cv2.CAP_PROP_FPS)
    nfrm = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # preprocessing
    ret, frame = video.read()
    roi = imrect(frame)
    bg = generate_background(video, roi, 100)

    # analysis
    print('Calculating fly positions', end='')
    global search_size
    threshold = search_size ** 2 * per_pixel_threshold
    search_size = round(search_size / 2)

    # extract positions from video
    positions = np.zeros((nfrm, 3))
    positions[:, 0] = np.arange(0, (nfrm / framerate), 1 / framerate)  # time in seconds since video start
    xmax = roi[0] + roi[2]
    ymax = roi[1] + roi[3]
    for frm in range(0, nfrm):
        print('.', end='')
        status, frame = video.read(frm)
        fg = frame[roi[1]:ymax, roi[0]:xmax].mean(axis=2)
        fg = (256 * fg) / (bg + 1)
        positions[frm, 1:2] = findfly(fg, threshold, search_size, True)
    print('done!')

    video.release()


def findfly(image, threshold, size, invert):
    # Locate darkest pixel.
    if invert:
        val = image[:].min()
    else:
        val = image[:].max()
    ypos, xpos = np.nonzero(image == val)
    xpos = xpos.mean()
    ypos = ypos.mean()

    # crop to search area size
    leftEdge = int(xpos - size)
    rightEdge = int(xpos + size)
    topEdge = int(ypos - size)
    bottomEdge = int(ypos + size)
    if leftEdge < 0:
        leftEdge = 0
    if rightEdge > len(image[0, :]):
        rightEdge = len(image[0, :]) - 1
    if topEdge < 0:
        topEdge = 0
    if bottomEdge > len(image[:, 0]):
        bottomEdge = len(image[:, 0]) - 1
    area = image[topEdge:bottomEdge, leftEdge:rightEdge]

    # "Flip" image to be white pixels on black.
    if invert:
        area = 255 - area

    total = area.sum()
    if total < threshold:
        # do not return a valid position, likely an artifact
        return [np.nan, np.nan]
    else:
        x2 = np.arange(0, len(area[0, :]), 1)
        y2 = np.arange(0, len(area[:, 0]), 1)
        x = (area * x2).sum(axis=0) / total + leftEdge
        y = (area * y2).sum(axis=1) / total + topEdge
        return [x, y]




def imrect(image):
    """
    Best python hack of MATLAB's imrect that I could come up with

    :param image: a numpy array
    :return: returns the bounds of the selected roi
    """
    print('Click and drag to select ROI. Press "Enter" to proceed.')
    fig = plt.figure('ROI selection')
    plt.imshow(image)
    ax = plt.axes()

    def nothing(eclick, erelease):
        pass

    bounds = None
    def on_enter(event):
        if event.key == 'enter' and selector._rect_bbox != (0., 0., 0., 1.):
            nonlocal bounds
            bounds = [int(i) for i in selector._rect_bbox]
            plt.close(fig)

    selector = widgets.RectangleSelector(ax, nothing, drawtype='box', interactive=True)
    plt.connect('key_press_event', on_enter)
    plt.show()  # blocks execution until on_enter() completes
    return bounds


def generate_background(vreader, bounds, depth):
    """
    Create a background image for use in subtraction

    :param vreader: an instance of cv2.VideoCapture()
    :param bounds: bounds to crop the image with
    :param depth: number of frames to average
    :return: an averaged greyscale frame (as a numpy array)
    """
    print('Creating background', end='')
    randv = [random.randint(0, vreader.get(cv2.CAP_PROP_FRAME_COUNT - 1)) for i in range(0, depth)]
    bg_array = np.zeros((bounds[3], bounds[2], depth), dtype='uint8')
    for frame in range(0, 100):
        print('.', end='')
        status, img = vreader.read(randv[frame])
        plt.show()
        bg_array[:, :, frame] = img[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2], :].mean(axis=2)
    print('done!')
    return bg_array.mean(axis=2)

if __name__=='__main__':
    main(sys.argv)