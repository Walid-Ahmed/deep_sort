# from urllib2 import urlopen, URLError, HTTPError

from urllib.request import urlopen, URLopener
from console_progressbar import ProgressBar

import os
import zipfile
import json
import sys
# import urlib

def download_progress(t_blocks, block_size, file_size):
    if file_size < 0:
        file_size = ((t_blocks + 1) * block_size)

    pb.print_progress_bar((t_blocks * block_size) / file_size * 100)


def unzipfile(file):
    try:
        zip_ref = zipfile.ZipFile(file, 'r')
        # make directory with file name
        # os.mkdir(file.split('.')[0])
        zip_ref.extractall("./")
        zip_ref.close()   
    except Exception as e:
        pass  
    

def dlfile(basename, url, filename):
    # Open the url
    # filealready exits
    file_path = os.path.join(os.getcwd(), basename + filename)
    exists = os.path.isfile(file_path)
    if exists:
        pass
    else:
        opener.retrieve(url, file_path, download_progress)
    
    print("\nDownload Complete\n")
    
    if filename.endswith('.zip'):
        print("\nUnzipping file...\n")
        unzipfile(filename)
        print("\nUnzipping Complete\n")




# os.path.join(os.getcwd(), basename + filename)
items = [("resources/networks/", "mars-small128.ckpt-68577", "http://download1644.mediafire.com/hl18j2gd9ueg/i8ulgnq050k8c9v/mars-small128.ckpt-68577"), 
    ("resources/networks/", "mars-small128.ckpt-68577.meta", "http://download846.mediafire.com/xtxd4zy8ephg/m7eciqc1q4ipi5v/mars-small128.ckpt-68577.meta"), 
    ("resources/networks/", "mars-small128.pb", "http://download2269.mediafire.com/aj1nq6dj3dfg/lch8dhv54obckb2/mars-small128.pb"), 
    ("./", "MOT16.zip", "https://motchallenge.net/data/MOT16.zip")]


opener = URLopener()
pb = ProgressBar(total=100,prefix='Here', suffix='Now', decimals=3, length=50, fill='X', zfill='-')

with open('app.config') as data:
    config = json.load(data)

LEFT_STR = '\n=========================================\n\t'
RGHT_STR = '\n=========================================\n'


os.system("clear")
os.system("echo '{}Downloading Requried Files....{}'".format(LEFT_STR, RGHT_STR))
for (basename, filename, url) in items:
    dlfile(basename, url, filename)



if config["mode"] != "online":
    os.system("clear")
    os.system("echo '{}Build Detection....{}'".format(LEFT_STR, RGHT_STR))
    os.system('python build_npy.py')


if config["play_demo"]:
    config["play_demo"] = False
    with open('app.config', 'w') as outfile:
        json.dump(config, outfile)

    os.system("clear")
    os.system("echo '{}Running Demo....{}'".format(LEFT_STR, RGHT_STR))
    os.system('python run.py')

