import cv2
import os
import pandas as pd
import numpy as np


folderOfDetections="fisheye03-20181127-104501-111504"
folderOfDetections="./MOT16/train/MOT16-13"

pathToDetectionFile=folderOfDetections+"/det/det.txt"
pathtoImages=folderOfDetections+"/img1"
fileName="demo_"+os.path.basename(folderOfDetections)+".mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
width,height=960,540

def readDetections():
	# reading csv file    #<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
	dataframe = pd.read_csv(pathToDetectionFile, delim_whitespace=False,header=-1,skiprows=0)
	print("Detection Data read from file  {}".format(pathToDetectionFile))
	dataset = dataframe.values
	frameNums=(dataset[:,0]).astype(int)
	objectIDs=(dataset[:,1]).astype(int)
	boundingBoxes=(dataset[:,2:6]).astype(int)    #<bb_left>, <bb_top>, <bb_width>, <bb_height>
	confs=dataset[:,6] 

	return dataset
	
	
def run(dataset):
	video_creator = cv2.VideoWriter(fileName,fourcc, 5, (width,height))
	print("Saving demo to file {0}".format(fileName))
	#exit()
	imgFiles=os.listdir(pathtoImages)
	imgFiles.sort()
	frameNum=1
	for filename in imgFiles:
		
		frameNum = int(filename.split('.')[0])
		print((filename, frameNum))
		img = cv2.imread(os.path.join(pathtoImages,filename))
		
		frameDetection=dataset[np.where(dataset[:,0] == frameNum)]
		boundingBoxes=(frameDetection[:,2:6]).astype(int) 
		# print(boundingBoxes)  #<bb_left>, <bb_top>, <bb_width>, <bb_height>
		for box in boundingBoxes:
			bb_left, bb_top, bb_width, bb_height=box
			xmin,ymin,xmax,ymax=bb_left, bb_top, bb_left+bb_width, bb_top+bb_height
			# ymin,xmin,xmax,ymax=bb_left, bb_top, bb_left+bb_width, bb_top+bb_height

			# xmin, ymin, xmax, ymax = bb_left, 1920 - bb_top - bb_width, bb_left+bb_height, 1920-bb_top
			
			cv2.rectangle(img,(ymin, xmin),(xmax, ymax),(0,255,0),1)
		if (img is None):
			frameNum=frameNum+1
			continue
		img=cv2.resize(img,(width, height))
		video_creator.write(img)
		cv2.imshow('indus.ai',img)
		cv2.waitKey(100)
		frameNum=frameNum+1
		print(frameNum)
		if cv2.waitKey(1) & 0xFF == ord('q'):
					break
    	 		
if os.path.exists(fileName):
  os.remove(fileName)
dataset=readDetections()
run(dataset)