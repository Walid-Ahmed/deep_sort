import os
import json


os.system("echo 'running....'")
advance_count = 1


with open('app.config') as data:
    config = json.load(data)
    print(config)
# before running the file make sure:
# make sure that --model points to the correct network file such as mars-small128.pb 
# make sure that --sequence_dir points to the sequence images such as MOT16-13 and it associated detections in the next flag
# make sure that --detection_file points to the correct npy file
config["mode"] = "deep_sort_app.py"
if config["mode"] == "online":
	config["mode"] = "deep_sort_app_online.py"

show = "python {} \
    --sequence_dir={} \
    --detection_file={} \
    --output_file={} \
    --min_confidence={} \
    --nn_budget={} \
    --display={} \
    --record_file={}".format(config["mode"], config["sequence_dir"], config["detection_file"], config["output_file"], config["min_confidence"], config["nn_budget"], config["display"], config["record_file"])

os.system(show)


