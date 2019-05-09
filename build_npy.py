import os
import json

os.system("echo 'running....'")
advance_count = 1


with open('app.config') as data:
    config = json.load(data)
    # print(config)


# gen_detection_cmd = "python tools/generate_detections.py --model=resources/networks/mars-small128.pb \
# --mot_dir=./MOT16/train \
# --output_dir=./resources/detections/MOT16_train"


gen_detection_cmd = "python tools/generate_detections.py --model={} \
--mot_dir={} \
--output_dir={}".format(config["model"], config["mot_dir"], config["output_dir"])




os.system(gen_detection_cmd)