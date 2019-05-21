# from urllib2 import urlopen, URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import urlretrieve, urlopen, build_opener, install_opener, URLopener, HTTPCookieProcessor
from console_progressbar import ProgressBar
from os.path import splitext, basename
from bs4 import BeautifulSoup

import os
import tarfile
import zipfile
import json
import sys
import shutil
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
    

def parse_page(link):
    if 'mediafire' not in link:
        return link

    response = urlopen(link).read()
    h = BeautifulSoup(response, 'html.parser').find('a', attrs={'class': 'input', 'aria-label': 'Download file'}).get('href')
    return h

def download_file(bn, url, filename):
    
    download_link = parse_page(url)
    url_components = urlparse(download_link)    
    name, file_ext = splitext(basename(url_components.path))
    print("Downloading {}".format(name+file_ext))
    file_path = os.path.join(os.getcwd(), os.path.join(bn, name) + file_ext)
    
    
    if os.path.isfile(file_path):
        pass
    else:
        # opener.retrieve(url, file_path, download_progress)
        print(download_link)
        urlretrieve(download_link, file_path, download_progress)
    
    print("\nDownload Complete\n")
    
    if file_ext == '.gz':
        check_folder_exits(name, file_ext)
        # print("\nUnzipping file...\n")
        # unzipfile(filename)
        # print("\nUnzipping Complete\n")

    rmv_MACOSX()


def rmv_MACOSX():
    if os.path.exists("__MACOSX"):
        shutil.rmtree("__MACOSX", ignore_errors=True, onerror=None)

    if os.path.exists(".DS_Store"):
        os.remove(".DS_Store")

    # os.makedirs(target_folder)


def check_folder_exits(name, file_ext):
    print("\nUnzipping file...\n")
    # if not os.path.exists(os.path.join(os.getcwd(), os.path.join(config['networks_path'], 'frozen_inference_graph.pb'))):
    tar_file = tarfile.open(os.path.join(config['networks_path'], name+file_ext))
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.path.join(os.getcwd(), config['networks_path']))
        # unzipfile(name+file_ext)
    print("\nUnzipping Complete\n")


# first load the config file 
with open('app.config') as data:
    config = json.load(data)

# os.path.join(os.getcwd(), basename + filename)


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'


items = [
    (config["networks_path"], "mars-small128.ckpt-68577", "http://www.mediafire.com/file/i8ulgnq050k8c9v/mars-small128.ckpt-68577/file"), 
    (config["networks_path"], "mars-small128.ckpt-68577.meta", "http://www.mediafire.com/file/m7eciqc1q4ipi5v/mars-small128.ckpt-68577.meta/file"), 
    (config["networks_path"], "mars-small128.pb", "http://www.mediafire.com/file/lch8dhv54obckb2/mars-small128.pb/file"),
    (config["networks_path"], MODEL_NAME + ".pb", DOWNLOAD_BASE + MODEL_FILE)
]


# opener = URLopener()
# this fix 302 error 
opener = build_opener(HTTPCookieProcessor())
install_opener(opener)
pb = ProgressBar(total=100,prefix='Here', suffix='Now', decimals=3, length=50, fill='X', zfill='-')



LEFT_STR = '\n=========================================\n\t'
RGHT_STR = '\n=========================================\n'

if not os.path.exists(config["networks_path"]):
    os.makedirs(config["networks_path"])



os.system("clear")
os.system("echo '{}Downloading Requried Files....{}'".format(LEFT_STR, RGHT_STR))
for (bn, filename, url) in items:
    try:
        download_file(bn, url, filename)
    except Exception as e:
        print(e)
        

os.system("clear")
os.system("echo '{}Running Demo....{}'".format(LEFT_STR, RGHT_STR))
os.system('python run.py')


    

