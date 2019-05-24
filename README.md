
# Deep SORT: online

## Introduction

This repository contains code for an extended version *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
We extend the original [Deep SORT](https://github.com/nwojke/deep_sort) repo to enable online tracking and ease of use. 

Note: SORT links or associates objects that appear in consecutive frames. As a consequence, detected objects should be available beforehand or at least the detected objects in frames needed to link. This extension includes SSD network part of [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to detect and feed objects to the SORT network.

## Dependencies

The code is compatible with Python 3. The following dependencies are
needed to run the tracker:

* NumPy
* sklearn
* OpenCV

Feature generation requires TensorFlow (>= 1.0).  
Additional libraries to install are [console progress bar](https://pypi.org/project/console-progressbar/) and [beautifulsoup](https://pypi.org/project/beautifulsoup4/)

## Installation

First, clone the repository:
```
git clone https://github.com/Walid-Ahmed/deep_sort.git
cd deep_sort
```

### This part may not be required

Then, run preparations file which will download the required files (4 files), the neural networks will be placed under the folder 'networks' containing the deep SORT and SSD networks. However, this repo is self-contained only run ```prep.py``` if some files are corrupted. Ideally, it is required to run only once.

The default behavior of ```prep.py``` is to check if the files already existed, but they are straightforward checks which can result in some errors to overcome these issues deleting the corrupted file would enforce downloading this file. 


``` 
python prep.py 
```

### Now, let's rock and roll.

## Running the tracker
Use ```run.py``` to run the tracker it will run the sample.mp4 demo. 
```
python run.py
```

The ```run.py``` is a simple script to execute the modified version of deep SORT. It encapsulates the execution of ```deep_sort_app_only.py``` by feeding the appropriate parameters where it collects them from the ```app.config``` file.


Common entries of ```app.config``` are: 

  - input_video: the location of input video file.
  - detection_threshold: a value ranges between 0.0 and 1.0 where detected objects with a score (confidence) below this value SSD will discard them.
  - f_model: the location of feature model or deep SORT model.
  - d_model: the location of SSD model.
  - networks_path: folder location where all neural networks reside in.
  - record_video: the location of the video output (record) file.
  - frame_rate: record file frames per second, the same frame rate of the input video is used if this entry does not exist.
  - More info about the supported entries in ```app.config``` is found using ```python deep_sort_app_online.py -h```.



## More details

Please check the original [repo](https://github.com/nwojke/deep_sort) for more details
