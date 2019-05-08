import os

show = "python tools/generate_detections.py --model=resources/networks/mars-small128.pb \
--mot_dir=./MOT16/train \
--output_dir=./resources/detections/MOT16_train"

os.system(show)