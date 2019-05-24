import os
import json


os.system("echo 'running....'")
advance_count = 1


with open('app.config') as data:
    config = json.load(data)
# before running the file make sure:
# make sure that --model points to the correct network file such as mars-small128.pb 
# make sure that --sequence_dir points to the sequence images such as MOT16-13 and it associated detections in the next flag
# make sure that --detection_file points to the correct npy file


show = "python deep_sort_app_online.py \
	--model={}\
    --input_video={} \
    --frame_rate={}\
    --threshold={}\
    --min_confidence={} \
    --nn_budget={} \
    --display={} \
    --record_video={}".format(config["f_model"], config["input_video"], int(config['frame_rate']), float(config["detection_threshold"]), config["min_confidence"], config["nn_budget"], config["display"], config["record_video"])

os.system(show)


