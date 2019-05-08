
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


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
with open('app.config') as data:
    config = json.load(data)
    print(config)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = config['model']
threshold = float(config['threshold'])
ids = ast.literal_eval(config['ids'])
print(ids)

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = config['labels']

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph, sess):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def frame_detections(path, sess, frame_count, width, height, det):
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph, sess)

  # Visualization of the results of a detection.
  scores = output_dict['detection_scores']
  bboxes = output_dict['detection_boxes']
  dclses = output_dict['detection_classes']

  # open file
  i = 0
  # with open('detections', 'w') as det:
  for i in range(len(scores)):
    if scores[i] >= threshold and dclses[i] in ids:
      # bb_left, bb_top, bb_width, bb_height = width * bboxes[i][0], height * bboxes[i][1], width * abs(bboxes[i][0] - bboxes[i][2]), height * abs(bboxes[i][1] - bboxes[i][3])

      bb_top, bb_left, bb_width, bb_height = height * bboxes[i][0], width * bboxes[i][1], height * abs(bboxes[i][0] - bboxes[i][2]), width * abs(bboxes[i][1] - bboxes[i][3])

      # bb_left, bb_top = height * (1.0 - bboxes[i][1]), width * bboxes[i][0]
      # bb_width, bb_height = width * (bboxes[i][0] - bboxes[i][2]), height * (bboxes[i][1] - bboxes[i][3])
      # append to the file
      # det.write('x]n')
      det.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(str(frame_count).zfill(10), dclses[i], bb_left, bb_top, bb_width, bb_height, scores[i], -1, -1, -1))
      print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(str(frame_count).zfill(10), dclses[i], bb_left, bb_top, bb_width, bb_height, scores[i], -1, -1, -1))
      # close file


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

import shutil
if os.path.exists(folder_name):
  shutil.rmtree(folder_name, ignore_errors=True, onerror=None)
os.makedirs(folder_name)

os.makedirs(os.path.join(folder_name, 'img1'))
os.makedirs(os.path.join(folder_name, 'det'))
os.makedirs(os.path.join(folder_name, 'gt'))

WIDTH, HEIGHT = 3, 4

vidcap = cv2.VideoCapture(video_name)
width, height = int(vidcap.get(WIDTH)), int(vidcap.get(HEIGHT))
print((width, height))
frame_count = 0
det = open('{}/det/det.txt'.format(folder_name), 'w')
with detection_graph.as_default():
  with tf.Session() as sess:

    while(vidcap.isOpened()):
      ret, image_np = vidcap.read()
      path = '{},{},{}.jpg'.format(folder_name, 'img1', str(frame_count).zfill(10)).split(',')
      cv2.imwrite(os.path.join(*path), image_np)
      frame_detections(None, sess, frame_count, width, height, det)
      # read_img(image_np, sess)
      if config['display']:
        cv2.imshow('object_detection', cv2.resize(image_np, (width, height)))
      # this is optional to quit in the middel
      frame_count += 1
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

    




