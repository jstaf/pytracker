#!/usr/bin/env python3
# we are recreating the old matlab fly tracker in python

import cv2
import sys
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

# configurable params
search_size = 20
per_pixel_threshold = 1.5
arena_size = 8 # assumes square arena

path = 'Aarya_videos/IMG_5807.m4v'
video = cv2.VideoCapture(path)
if not video.isOpened():
    sys.exit('Could not open: ' + path)
resolution = (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = video.get(cv2.CAP_PROP_FPS)
# last frame of the video
nfrm = int(video.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

# have a function to take user ROI input here
# unused for now, look into rectangle selection widget from matplotlib
# ret, frame = video.read()
# cv2.imshow('test', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# create a background from bg_depth random frames
bg_depth = 100
print('Creating background', end='')
randv = [random.randint(0, nfrm) for frame in range(0, bg_depth)]
bg_array = np.zeros((resolution[1], resolution[0], bg_depth), 'uint8')
for frame in range(0, 100):
    print('.', end='')
    status, gray = video.read(randv[frame])
    bg_array[:, :, frame] = gray.mean(axis=2)
bg = bg_array.mean(axis=2)
print('done!')

plt.imshow(bg, interpolation='nearest', cmap='Greys_r')
plt.colorbar()
plt.show()

# close reader
video.release()
