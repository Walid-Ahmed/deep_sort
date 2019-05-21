import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2
import os

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
# from object_detection.utils import ops as utils_ops

from tools.utils import label_map_util
# from utils import visualization_utils as vis_util


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'tools/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

class SSD(object):

	def __init__(self):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')
		self.sess = tf.Session(graph=self.detection_graph)

	# make sure to close the session    		
	def close(self):
		self.sess.close()


	def convert_output(self, output_dict, iframe, width, height, threshold):
		# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
		detections = []
		for i in range(len(output_dict['detection_boxes'])):
			if output_dict['detection_scores'][i] >= threshold:
				ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i] #box
				detections.append([iframe, -1, 
					xmin * width, 
					ymin * height, 
					abs(xmax-xmin) * width, 
					abs(ymax-ymin) * height, 
					output_dict['detection_scores'][i], -1, -1, -1])
		print(len(detections))
		return np.array(detections)

	def run_inference_for_single_image(self, image, image_index, threshold):
		with self.detection_graph.as_default():
			image_np_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
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
				    detection_masks, detection_boxes, image_np_expanded.shape[1], image_np_expanded.shape[2])
				detection_masks_reframed = tf.cast(
				    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
				    detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = self.sess.run(tensor_dict,
			                     feed_dict={image_tensor: image_np_expanded})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict[
			  'detection_classes'][0].astype(np.int64)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
				# filter output based on score threshold
			return self.convert_output(output_dict, image_index, image.shape[1], image.shape[0], threshold)

