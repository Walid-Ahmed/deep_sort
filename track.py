import json
import os

with open('track.config') as data:
    config = json.load(data)


# Track specific object
track = 'python show_results.py\
	--input_video={}\
	--object_id={}\
	--frame_rate={}\
	--output_file={}\
	'.format(config['input_video'], config['object_id'], config['frame_rate'], config['record_video'])
os.system(track)