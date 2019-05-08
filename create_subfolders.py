import numpy as np

video_name = 'the-bookshop-trailer-1_h480p.mov'
folder_name = video_name.split('.')[0]

import shutil, os
if os.path.exists(folder_name):
  shutil.rmtree(folder_name, ignore_errors=True, onerror=None)
os.makedirs(folder_name)

os.makedirs(os.path.join(folder_name, 'img1'))
os.makedirs(os.path.join(folder_name, 'det'))
os.makedirs(os.path.join(folder_name, 'gt'))


def frame_detections(path, sess, frame_count, width, height, det):
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  # image_np_expanded = np.expand_dims(image_np, axis=0)
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
    if scores[i] >= .5:
  	  bb_left, bb_top, bb_width, bb_height = width * bboxes[i][0], height * bboxes[i][1], width * (bboxes[i][0] - bboxes[i][2]), height * (bboxes[i][1] - bboxes[i][3])
  	  # append to the file
  	  # det.write('x]n')
  	  det.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(frame_count, dclses[i], bb_left, bb_top, bb_width, bb_height, scores[i], -1, -1, -1))
  	  print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(frame_count, dclses[i], bb_left, bb_top, bb_width, bb_height, scores[i], -1, -1, -1))
  	  # close file
  
import cv2


WIDTH, HEIGHT = 3, 4

vidcap = cv2.VideoCapture(video_name)
width, height = int(vidcap.get(WIDTH)), int(vidcap.get(HEIGHT))

frame_count = 0
det = open('{}/det/det.txt'.format(folder_name), 'w')
while(vidcap.isOpened()):
  ret, image_np = vidcap.read()
  path = '{},{},{}.jpg'.format(folder_name, 'img1', frame_count).split(',')
  cv2.imwrite(os.path.join(*path), image_np)
  frame_detections(None, None, frame_count, width, height, det)
  cv2.imshow('object_detection', cv2.resize(image_np, (width, height)))
  # this is optional to quit in the middel
  frame_count += 1
  if cv2.waitKey(25) & 0xFF == ord('q'):
    break
det.close()

