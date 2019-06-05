import os
import json

with open('app.config') as data:
    config = json.load(data)


os.system("echo '\nsaving video results'")
save_results_command = "python save_results.py --source_video={}".format(config['input_video'])
os.system(save_results_command)






