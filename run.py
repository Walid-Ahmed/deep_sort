import os
# before running the file make sure:
# make sure that --model points to the correct network file such as mars-small128.pb 
# make sure that --sequence_dir points to the sequence images such as MOT16-13 and it associated detections in the next flag
# make sure that --detection_file points to the correct npy file


show = "python deep_sort_app.py \
	--model=resources/networks/mars-small128.pb \
    --sequence_dir=./MOT16/train/MOT16-13 \
    --detection_file=./resources/detections/MOT16_train/MOT16-13.npy \
    --output_file=./resources/detections/MOT16_records/MOT16-13.txt \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True \
    --record_file=x.avi"
os.system(show)


