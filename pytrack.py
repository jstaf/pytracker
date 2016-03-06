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
arena_size = 8 # assumes square arena


def main(argv):
    path = 'Aarya_videos/IMG_5807.m4v'
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        sys.exit('Could not open: ' + path)
    resolution = (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = video.get(cv2.CAP_PROP_FPS)
    # last frame of the video
    nfrm = int(video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret, frame = video.read()
    roi = imrect(frame)

    # bg = generate_background(video, 100)
    # plt.imshow(bg, interpolation='nearest', cmap='magma')
    # plt.colorbar()
    # plt.show()

    # close reader
    video.release()


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
    plt.show()
    return bounds


def generate_background(vreader, depth):
    """
    Create a background image for use in subtraction
    :param vreader: an instance of cv2.VideoCapture()
    :param depth: number of frames to average
    :return: an averaged greyscale frame (as a numpy array)
    """
    print('Creating background', end='')
    randv = [random.randint(0, vreader.get(cv2.CAP_PROP_FRAME_COUNT - 1)) for i in range(0, depth)]
    bg_array = np.zeros((vreader.get(cv2.CAP_PROP_FRAME_HEIGHT), vreader.get(cv2.CAP_PROP_FRAME_WIDTH), depth), 'uint8')
    for frame in range(0, 100):
        print('.', end='')
        status, gray = vreader.read(randv[frame])
        bg_array[:, :, frame] = gray.mean(axis=2)
    print('done!')
    return bg_array.mean(axis=2)

if __name__=='__main__':
    main(sys.argv)