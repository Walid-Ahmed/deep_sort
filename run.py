import os
import json

from save_results import *


os.system("echo '\nrunning....'")
advance_count = 1


with open('app.config') as data:
    config = json.load(data)

online_sort_command = "python deep_sort_app_online.py \
	--frozen={}\
	--model={}\
    --input_video={} \
    --frame_rate={}\
    --threshold={}\
    --min_confidence={} \
    --nn_budget={} \
    --record_video={}".format(config['d_model'], config["f_model"], config["input_video"], int(config['frame_rate']), float(config["detection_threshold"]), config["min_confidence"], config["nn_budget"], config["record_video"])
os.system(online_sort_command)

os.system("echo '\nsaving video results'")
save_results_command = "python save_results.py --source_video={}".format(config['input_video'])
os.system(save_results_command)






