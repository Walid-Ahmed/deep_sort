
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import ast

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image



# What model to download.
with open('app.config') as data:
    config = json.load(data)
    print(config)


import cv2

video_name = config['name']
folder_name = video_name.split('.')[0]
WIDTH, HEIGHT = 3, 4

vidcap = cv2.VideoCapture(video_name)
width, height = int(vidcap.get(WIDTH)), int(vidcap.get(HEIGHT))


# create the folder structure

# video name
#   img1 folder
#   det folder
#   gt folder
#   seqinfo ini

# import shutil
# if os.path.exists(folder_name):
#   shutil.rmtree(folder_name, ignore_errors=True, onerror=None)
# os.makedirs(folder_name)

# os.makedirs(os.path.join(folder_name, 'img1'))
# os.makedirs(os.path.join(folder_name, 'det'))
# os.makedirs(os.path.join(folder_name, 'gt'))

WIDTH, HEIGHT = 3, 4

vidcap = cv2.VideoCapture(video_name)
width, height = int(vidcap.get(WIDTH)), int(vidcap.get(HEIGHT))

frame_count = 1

while(vidcap.isOpened()):
  ret, image_np = vidcap.read()
  path = '{},{},{}.jpg'.format(folder_name, 'img1', str(frame_count).zfill(10)).split(',')
  cv2.imwrite(os.path.join(*path), image_np)
  
  # this is optional to quit in the middel
  frame_count += 1
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
    
  if frame_count >= 1000:
    break

cv2.destroyAllWindows()

    




