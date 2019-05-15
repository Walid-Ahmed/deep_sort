# Deep SORT

## Introduction

This repository contains code for an extended version *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
We extend the original [Deep SORT](https://github.com/nwojke/deep_sort) repo to enable online tracking and simpler to use. 

Note: SORT links objects or associate that appear in consecutive frames. Detected objects should be available beforehand or at least the detected objects in the frames need to make association so if the word "detection(s)" is used in deep SORT context it probably means objects' association.

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
git clone https://github.com/Walid-Ahmed/deep_sort
cd deep_sort
```

Then, run preparations file which will download the required files, the neural network will be placed under the folder 'resources\networks\' and MOT challenge benchmark (MOT16) data as a zip file and will be extracted under MOT16 folder then object association files will be created. Finally, a demo will run for the first time to show the end result of Deep SORT algorithm. 

The default behavior of ```prep.py``` is to check if the files already existed but they are very simple checks which can result in some errors to overcome these issues two flags were added to the ```app.config``` which enforces downloading and enforce unzipping files or deleted the corrupted file which will also enforce downloading only for this file. 

These files are downloaded to show and experiment the deep SORT potentials. To use a custom dataset, executing ```prep.py``` should be ignored and the appropriate flags in ```app.config``` should be modified as well. Please check sections below for more details.

WARNINGS 
  - MOT data requires enough space on disk. 
  - If the download links are broken please download them manually from [here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp) and place them under the designated folder above. 
  - Uncompress the MOT16.zip file manually if it did unizip not automatically.
``` 
python prep.py 
```

### Now, let's rock and roll.

## Running the tracker
Simply use ```run.py``` to run the tracker by default it will run the demo displayed after running the ```prep.py```. 
```
python run.py
```

The ```run.py``` is a simple script to execute a modified version of ```deep_sort_app.py``` with appropriate parameters as well as the offline version. The online version ignores loading object association files (npy files) and instead loads the model, feed the images and display it as video as soon as the results computed which we refer to it as "online" version.  The ```run.py``` checks for the existence of the association file (npy file) in case of the offline execution which would result in an error otherwise it runs the online directly. The parameters are collected for the ```app.config``` file.

In addition, to run a different MOT sequence other than the demo, by modifying the ```app.config``` and pointing the ```sequence_dir``` entry to a different folder of the MOTs and also pointing ```detection_file``` entry to the corresponding ```npy``` file. The entries ```display``` and ```recored_file``` will display the sequence as video and record it as a video file, respectively. 

Common entries of ```app.config``` are: 

  - sequence_dir points to the sequence directory or other directory with similar structure, it contains the images and detected objects for each frame.
  - detection_file points to the ```npy``` file which cantains the association between two conscutive frames; it must be compiled or downloaded before running the offline version.
  - record_file points to the output video file to record the result.
  - res_struct entry is used to control the resurces folder structure in case of using the default settings.
  - More info about the supported entries in ```app.config``` is found using ```python deep_sort_app.py -h```.

Moreover, to run a custom video other than MOTs, there are two steps to follow [Create sequence directory](#Create-sequence-directory) similar to the MOT sequence folder structure and [Generate objects associations](#Generate-objects-associations). But please note that **videos are not single files such as mp4 or avi instead videos are a sequence of frames which referred to as a sequence directory**. 

## Create sequence directory

A MOT sequence follows standard rules avaliable on the [MOT challenge web page](https://motchallenge.net). In summary, there are three main folders and a file. 
The folder structure contains the following:

  - img1 a folder which contains the images for tracking multiple objects
  - det a folder which contains objects positions in each frame
  - gt a folder which contain the ground truth values
  - seqinfo.ini a file which is contains some information such as frame rate

More details are found on [MOT challenge web page](https://motchallenge.net)


## Generate objects associations

There is an additional utility that is used to simplyfy object association generations (npy files) which is ```build_npy.py```. The utility uses the same ```app.config``` to point some additional entries. 

  - model points to the neural network model path.
  - mot_dir which points to the parent folder of the MOT sequence directories. (not the MOT sequence itself) 
  - output_dir which points to the output folder where npy files are placed.

WARNING: do not confuse between mot_dir and sequence_dir and this process may take some time


## Modifications overview

The frames are fed one by one to the network and each frame is displayed as soon as the objects' association are computed. This modification saves the npy compile (the association between objects in consecutive frames) time and it is faster to experiment but the images are displayed a bit slower, however, it reflects the actual or near the actual time when deep SORT is deployed.

The goal is to process frames and displaying them in the same step instead of two separate ones (build the association for the full sequence then display it). The modification loads the model, process frames, bypass the npy compilation, and display the frame as soon as it processed. The modified version deep_sort_app was created to enable such execution. The ```deep_sort_app_online.py``` runs very similarly to the previous version with roughly the same arguments such as the SORT model and sequence directory paths.

Moreover, to encapsulate the details of running ```deep_sort_app_online.py``` and ```deep_sort_app.py```, the ```run.py``` script was added to simplify the execution and switching between the two versions by modifying the ```app.config``` attributes such as ```mode``` to select a version to run (online/offline) or ```sequence_dir``` to change the sequence directory.


## More details

Please check the original [repo](https://github.com/nwojke/deep_sort) for more details

