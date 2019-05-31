# vim: expandtab:ts=4:sw=4
import argparse

import cv2
import numpy as np

from deep_sort.iou_matching import iou
from application_util import visualization


DEFAULT_UPDATE_MS = 20


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


def run(vcap, tk_ids, f_rate, update_ms=None, video_filename=None):
    """Run tracking result visualization.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    result_file : str
        Path to the tracking output file in MOTChallenge ground truth format.
    show_false_alarms : Optional[bool]
        If True, false alarms are highlighted as red boxes.
    detection_file : Optional[str]
        Path to the detection file.
    update_ms : Optional[int]
        Number of milliseconds between cosecutive frames. Defaults to (a) the
        frame rate specifid in the seqinfo.ini file or DEFAULT_UDPATE_MS ms if
        seqinfo.ini is not available.
    video_filename : Optional[Str]
        If not None, a video of the tracking results is written to this file.

    """

    seq_info = gather_video_info(vcap, f_rate)
    results = np.loadtxt('Tracking_Results/tr.csv', delimiter=',')
    
    def frame_callback(vis, frame_idx):
        print("Frame idx", frame_idx)

        image = get_frame(vcap, frame_idx)
        vis.set_image(image.copy())

        # mask = np.logical_and(results[:, 0].astype(np.int) == frame_idx, results[:, 1].astype(np.int) == tk_id)
        mask = np.logical_and(results[:, 0].astype(np.int) == frame_idx, np.isin(results[:, 1].astype(np.int), np.array(tk_ids)))
        track_ids = results[mask, 1].astype(np.int)
        boxes = results[mask, 2:6]
        vis.draw_groundtruth(track_ids, boxes)

    # if update_ms is None:
    #     update_ms = seq_info["update_ms"]
    if update_ms is None:
        update_ms = DEFAULT_UPDATE_MS
    visualizer = visualization.Visualization(seq_info, update_ms)

    if video_filename is not None:
        visualizer.viewer.enable_videowriter(output_filename=video_filename, fps=f_rate)    
    visualizer.run(frame_callback)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--input_video", help="Path to input video.",
        default=None, required=True)
    parser.add_argument(
        "--object_ids", help="object id to track on video.",
        nargs='+', type=int, default=None)
    parser.add_argument(
        "--frame_rate", help="fps for the output video.",
        type=int, default=None)
    parser.add_argument(
        "--update_ms", help="",
        default=None)
    parser.add_argument(
        "--output_file", help="Filename of the (optional) output video.",
        default=None)
    
    return parser.parse_args()



args = parse_args()

# load input video
vcap = cv2.VideoCapture(args.input_video)
f_rate = args.frame_rate
if not args.frame_rate:
    f_rate = vcap.get(cv2.CAP_PROP_FPS)
# print(args.object_ids)
# # print(np.array(args.object_ids, dtype=int))
run(vcap=vcap, tk_ids=args.object_ids, f_rate=f_rate, update_ms=args.update_ms, video_filename=args.output_file)
    
