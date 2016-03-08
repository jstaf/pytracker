#!/usr/bin/env python3
# we are recreating the old matlab fly tracker in python

import cv2
import sys
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import widgets
from matplotlib.collections import LineCollection
import scipy.spatial.distance as dist

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
    for frm in range(0, nfrm - 1):
        if frm % 100 == 0:
            print('\n')
            print(frm, end='', flush=True)
        status, frame = video.read(frm)
        if not status:
            print('x', end='', flush=True)
            positions[frm, 1:] = [np.nan, np.nan]
        else:
            print('.', end='', flush=True)
            fg = frame[roi[1]:ymax, roi[0]:xmax].mean(axis=2)
            fg = (256 * fg) / (bg + 1)
            positions[frm, 1:] = findfly(fg, threshold, search_size, True)
    print('done!')
    video.release()

    print('Creating output')
    # convert coordinates to real world measurements
    positions[:, 1] = positions[:, 1] * (arena_size / roi[2])
    positions[:, 2] = positions[:, 2] * (arena_size / roi[3])
    # clean data
    positions = dist_filter(positions, 2)
    positions = interpolate(positions, 2)
    np.savetxt('positions.csv', positions, fmt='%10.5f', delimiter=',')
    plot_positions(positions, arena_size)


def interpolate(array, distance):
    """
    Linearly interpolate missing values between points as long as the subject does not move beyond a certain distance.

    :param array: A numpy array of positions
    :param distance: Distance beyond which interpolation is not performed
    :return: The "corrected" array
    """

    inum = 0
    for dim in range(1, array.shape[1], 2):
        npoint = 0
        while npoint < array.size - 1:
            point = array[npoint, dim]
            # if an nan is encountered, find the last and next point that was not an nan
            if np.isnan(point) and npoint != 1:
                lidx = npoint - 1
                lpoint = array[lidx, dim:(dim + 2)]
                nidx = not np.isnan(array[npoint:, dim]) + npoint - 1
                if np.isnan(nidx):
                    break
                npoint = array[nidx, dim:(dim + 2)]
                diff = nidx - npoint

                # make sure values aren't too far apart, then interpolate
                if dist.euclidean(lpoint, npoint) <= distance:
                    for badidx in range(1, diff):
                        array[lidx + badidx, dim:(dim + 2)] = lpoint + ((npoint - lpoint) * badidx / (diff + 1))
                    inum += diff

                # skip to next non-nan value
                npoint = nidx
    return array


def dist_filter(array, distance):
    """
    Iterate through array and remove spurious tracks

    :param array: A numpy array of positions (same format as the plotting function)
    :param distance: Movement greater than this distance is removed
    :return: The filtered array
    """
    navg = 5
    nfilt = 0
    for dim in range(1, array.shape[1], 2):
        for npoint in range(navg, array.shape[0] - navg):
            point = array[npoint, dim:(dim + 2)]
            if np.isnan(point[0]):
                continue
            else:
                # Compute mean positions for last and next num_avg frames
                last_set = array[(npoint - navg):npoint, dim:(dim + 2)]
                last_set = last_set[np.invert(np.isnan(last_set[:, 0])), :]  # wow, numpy is awkward to use
                last_mean = last_set.mean(axis=0)
                next_set = array[(npoint + 1):(npoint + 6), dim:(dim + 2)]
                next_set = next_set[np.invert(np.isnan(next_set[:, 0])), :]  # wow, numpy is awkward to use
                next_mean = next_set.mean(axis=0)

                # If the tracks move more than the threshold, erase it
                if (not np.isnan(last_mean[0]) and dist.euclidean(point, last_mean) > distance) or \
                        (not np.isnan(next_mean[0]) and dist.euclidean(point, next_mean) > distance):
                    array[npoint, dim:(dim + 2)] = np.nan
                    nfilt += 1

    print(nfilt, ' false tracks removed from the dataset.')
    return array


def plot_positions(dat, size):
    """
    Self explanatory, plots fly position.

    :param dat: An Nx3 array of positions where [time, xpos, ypos]
    :param size: Arena size
    :return:
    """
    points = dat[:, 1:].reshape((-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt.get_cmap('viridis'), norm=plt.Normalize(0, max(dat[:, 0])))
    lc.set_array(dat[:, 0])
    lc.set_linewidth(2)

    plt.figure('Pathing map')
    plt.gca().add_collection(lc)
    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()


def findfly(image, threshold, size, invert):
    """
    Locate a fly by finding the lightest (default)/darkest pixel, extracting the area around that pixel, then finding
    its centroid.

    :param image: A numpy array
    :param threshold: A per-pixel threshold
    :param size: Size of area in which to calculate the centroid
    :param invert: Whether to invert the region to find dark on light objects.
    :return: [x, y] coordinate of fly
    """
    # Locate darkest pixel.
    if invert:
        val = image[:].min()
    else:
        val = image[:].max()
    ypos, xpos = np.nonzero(image == val)
    xpos = xpos.mean()
    ypos = ypos.mean()

    # crop to search area size
    left_edge = int(xpos - size)
    right_edge = int(xpos + size)
    top_edge = int(ypos - size)
    bottom_edge = int(ypos + size)
    if left_edge < 0:
        left_edge = 0
    if right_edge > len(image[0, :]):
        right_edge = len(image[0, :]) - 1
    if top_edge < 0:
        top_edge = 0
    if bottom_edge > len(image[:, 0]):
        bottom_edge = len(image[:, 0]) - 1
    area = image[top_edge:bottom_edge, left_edge:right_edge]

    # "Flip" image to be white pixels on black.
    if invert:
        area = area.max() - area

    total = area.sum()
    if total < threshold:
        # do not return a valid position, likely an artifact
        return [np.nan, np.nan]
    else:
        xweight = np.arange(0, len(area[0, :]), 1)
        yweight = np.arange(0, len(area[:, 0]), 1)
        x = (area * xweight).sum() / total + left_edge
        y = (area.transpose() * yweight).sum() / total + top_edge
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
        print('.', end='', flush=True)
        status, img = vreader.read(randv[frame])
        plt.show()
        bg_array[:, :, frame] = img[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2], :].mean(axis=2)
    print('done!')
    return bg_array.mean(axis=2)

if __name__=='__main__':
    # main(sys.argv)

    testd = 1
    positions = np.loadtxt('positions.csv', delimiter=',')
    positions = dist_filter(positions, testd)
    positions = interpolate(positions, testd)
    plot_positions(positions, arena_size)