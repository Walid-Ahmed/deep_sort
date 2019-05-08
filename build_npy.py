import os

gen_detection_cmd = "python tools/generate_detections.py --model=resources/networks/mars-small128.pb \
--mot_dir=./MOT16/train \
--output_dir=./resources/detections/MOT16_train"

os.system(gen_detection_cmd)