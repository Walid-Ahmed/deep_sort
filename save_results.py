# vim: expandtab:ts=4:sw=4
import argparse

import cv2
import numpy as np
import colorsys

from deep_sort.iou_matching import iou
from application_util import visualization


DEFAULT_UPDATE_MS = 20
HEIGHT = 3
WIDTH = 4
FRAMES_NUM = 7


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


def rectangle(x, y, w, h, image, color, thickness=2, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(image, pt1, pt2, color, thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                text_size[0][1]
            cv2.rectangle(image, pt1, pt2, color, -1)
            cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), thickness)

def draw_groundtruth(track_id, box, img):
    thickness = 2
    color = create_unique_color_uchar(track_id)
    rectangle(*box.astype(np.int), img, color, thickness, label=str(track_id))

def write_video_with_object_i(orig_video_path, object_i, tracking_of_object_i, fourcc_string='MJPG', fps=24):
    vcap = cv2.VideoCapture(orig_video_path)
    img_size = (int(vcap.get(HEIGHT)), int(vcap.get(WIDTH)))
    fourcc = cv2.VideoWriter_fourcc(*fourcc_string) # Be sure to use lower case

    vwriter = cv2.VideoWriter('Tracking_Results/{}/{}.avi'.format(object_i, object_i), 
        fourcc, fps,  img_size)
    frame_idx = 0
    frames_of_object_i = tracking_of_object_i[:, 0]
    while True:
        ret, img = vcap.read()
        if not ret:
            vcap.release()
            vwriter.release()
            break


        if frame_idx in frames_of_object_i:
            frame_mask = tracking_of_object_i[:, 0].astype(np.int) == frame_idx
            box = tracking_of_object_i[frame_mask, 2:6][0]
            # print(box)
            draw_groundtruth(object_i, box, img)

        # write image to the video
        vwriter.write(cv2.resize(img.copy(), img_size))
        frame_idx += 1


def save_video_results(orig_video_path):
    results = np.loadtxt('Tracking_Results/tr.csv', delimiter=',')
    # select all objects unique
    ids = np.unique(results[:, 1].astype(int))
    for i in ids:
        mask_of_object_i = results[:, 1] == i
        tracking_of_object_i = results[mask_of_object_i, :]
        write_video_with_object_i(orig_video_path, i, tracking_of_object_i)

    


