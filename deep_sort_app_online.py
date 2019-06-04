# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import shutil

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools.generate_detections import * 
from save_results import *

HEIGHT = 3
WIDTH = 4
FRAMES_NUM = 7
num_objects = 0

def get_frame(vcap, frame_num):
    vcap.set(1, frame_num)
    return vcap.read()[1] # extract spcific frame from a video


# load display options from the current video like frame rates, image size.
def gather_video_info(vcap, f_rate):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available)."""    
   
    # if vcap:
    #     update_ms = 1000/ int(vcap.get(cv2.CAP_PROP_FPS))
    # else:
    #     update_ms = None
    max_frame_idx = int(vcap.get(FRAMES_NUM)) - 1

    image_size = (vcap.get(WIDTH), vcap.get(HEIGHT))
    # feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": str(vcap),
        "image_size": image_size,
        "min_frame_idx": 0,
        "max_frame_idx": max_frame_idx,
        "update_ms": f_rate
    }
    return seq_info



def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    # frame_indices = detection_mat[:, 0].astype(np.int)
    # mask = frame_indices == frame_idx

    
    # if there is no detections at all
    if not detection_mat:
        return []

    # otherwise include features for each detected object
    detection_list = []
    for row in detection_mat:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(frozen_model_path, model, vcap, f_rate, threshold, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, record_file): # I should add model here as an argument

    
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    # load detection model 
    load_model(frozen_model_path)
    seq_info = gather_video_info(vcap, f_rate)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    # delete tracking results if exists before running new video
    if os.path.exists('Tracking_Results'):
        shutil.rmtree('Tracking_Results', ignore_errors=True, onerror=None)
    

    def frame_callback(vis, frame_idx):

        def save_object():
            objects_path = 'Tracking_Results/{}'
            if not os.path.exists(objects_path.format(track.track_id)):
                os.makedirs(objects_path.format(track.track_id))
            
            global num_objects
            # cv2.imwrite(os.path.join(objects_path.format(track.track_id), 
            #     '{}_{}_{}.jpg'.format(frame_idx, track.track_id, num_objects)), 
            #     image[int(miny): int(maxy), int(minx): int(maxx), :])
            cv2.imwrite(os.path.join(objects_path.format(track.track_id), 
                '{}.jpg'.format(str(frame_idx).zfill(10))), 
                image[int(miny): int(maxy), int(minx): int(maxx), :])
            num_objects += 1

        print("Processing frame %05d" % frame_idx)
        # Load image and generate detections.
        # this is the part where to make online, get a detection for a single frame
        # create a new method for frame call back that call the detection network and give back
        # detections = create_detections(
        #     seq_info["detections"], frame_idx, min_detection_height)

        # enable online detections
        # detections = get_detections(model, frame_idx, sequence_dir)
        detections = create_detections(get_detections2(model, get_frame(vcap, frame_idx), frame_idx, threshold), 
            frame_idx, min_detection_height)
        
        detections = [d for d in detections if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        image = get_frame(vcap, frame_idx)

        # if display:
            #image = get_frame(vcap, frame_idx)
        vis.set_image(image.copy())
        vis.draw_detections(detections)
        vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])


            # saving detected objects
            minx, miny, maxx, maxy = track.to_tlbr()
            save_object()

        
            

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info, update_ms=5)

    visualizer.viewer.enable_videowriter(output_filename=record_file, fps=f_rate)
    visualizer.run(frame_callback)
    vcap.release()
    # Store results.

    # f = open(output_file, 'w')
    f = open('Tracking_Results/tr.csv', 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)



def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--frozen", help="Path to detection model",
        default=None, required=True)

    parser.add_argument(
        "--model", help="Path to deep sort model",
        default=None, required=True)
    parser.add_argument(
        "--threshold", help="Every detection below this value will be ignored",
        default=0.5, type=float, required=False)
    parser.add_argument(
        "--input_video", help="Path to the input video file.",
        default=None, required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.csv")
    parser.add_argument(
        "--frame_rate", help="Record file's number of frames per second",
        default=None, required=False)
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    parser.add_argument(
        "--record_video", help="Enter record file name",
        default='x.mp4', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load the model here
    encoder = create_box_encoder(args.model, batch_size=1)
    # load input video
    vcap = cv2.VideoCapture(args.input_video)
    f_rate = args.frame_rate
    if not args.frame_rate:
        f_rate = vcap.get(cv2.CAP_PROP_FPS)

    run(args.frozen, encoder, vcap, int(f_rate), args.threshold, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display, args.record_video)

    

    


