# Deep SORT

## Introduction

This repository contains code for an extended version *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
We extend the original [Deep SORT](https://github.com/nwojke/deep_sort) repo to enable online tracking and simpler to use.

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
git clone https://github.com/nwojke/deep_sort.git
cd deep_sort
```

Then, run preparations file which will download the required files, the neural network will be placed under the folder 'resources\networks\' and MOT challenge benchmark (MOT16) data as a zip file and will be extracted under MOT16 folder then detection files will be created. Finally, a demo will run for the first time to show the end result of Deep SORT algorithm. The default behavior of ```prep.py``` is to check if the files already existed but they are very simple checks which can result in some errors to overcome these issues two flags were added to the ```app.config``` which enforces downloading and enforce unzipping files or deleted the corrupted file which will also enforce downloading but only for this file. These files are downloaded to show and experiment the deep SORT potentials and to use custom dataset, executing ```prep.py``` should be ignored and the appropriate flags in ```app.config``` should be modified as well. Please check sections below for more details.

WARNING: motdata requires enough space on disk, in addition, if the download links are broken please download them manually from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) place them under the designated folder above.
``` 
python prep.py 
```

### Now, let's rock and roll.

## Running the tracker
Simply use ```run.py``` to run the tracker by default it will run the demo displayed after running the ```prep.py```. 
```
python run.py
```

```run.py``` is a simple script to execute a modified version of ```deep_sort_app.py``` with appropriate parameters as well as the offline version where it ignores loading detections files (npy files) and instead loads the model, feed the images and display it as video as soon as the results computed which we refer to it as "online" version.  The ```run.py``` checks for the existence of the detection file (npy file) in case of the offline execution which would result in an error otherwise it runs the online directly. The parameters are collected for the ```app.config``` file.

In addition, to run a different MOT sequence other than the demo, by modifying the ```app.config``` and pointing the ```sequence_dir``` entry to a different folder of the MOTs and also pointing ```detection_file``` entry to the corresponding ```npy``` file. The entries ```display``` and ```recored_file``` will display the sequence as video and record it as a video file, respectively. 

Common entries of ```app.config``` are: 

  - --sequence_dir points to the sequence director or other director with similar structure
  - --detection_file points to the ```npy``` file that corresponds to the previous sequence 
  - --record_file points to the output file to record the result

Moreover, to run a custom sequence other than MOTs, there are two steps to follow [Create sequence directory](#Create-sequence-directory) similar to MOT sequence folder structure and [Generate detections](#Generating-detections). 

## Create sequence directory

A MOT sequence follows standard rules avaliable on the [MOT challenge web page](https://motchallenge.net). In summary, there are three main folders and a file. 
The folder structure contains the following:

  - img1 a folder which contains the images for tracking multiple objects
  - det a folder which contains object positions in each frame
  - gt a folder which contain the ground truth values
  - seqinfo.ini a file which is contains some information such as frame rate

More details are found on [MOT challenge web page](https://motchallenge.net)


## Generating detections

There is an additional utility that is used to simplyfy detections generations (npy files) which is ```build_npy.py```. The utility uses the same ```app.config``` to point some additional entries. 

  - --model points to the neural network
  - --mot_dir which points to the parent folder of the MOT sequence directory (not the MOT sequence itself) 
  - --output_dir which points to the output folder

WARNING: do not confuse between mot_dir and sequence_dir and this process may take some time


## More details

please check the original [repo](https://github.com/nwojke/deep_sort) for more details