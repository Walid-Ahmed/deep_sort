# from urllib2 import urlopen, URLError, HTTPError

from urllib.request import urlopen, URLopener
from console_progressbar import ProgressBar

import os
import zipfile
# import urlib

def download_progress(t_blocks, block_size, file_size):
    if file_size < 0:
        file_size = ((t_blocks + 1) * block_size)

    pb.print_progress_bar((t_blocks * block_size) / file_size * 100)


def unzipfile(file):
    zip_ref = zipfile.ZipFile(file, 'r')
    # make directory with file name
    os.mkdir(file.split('.')[0])
    zip_ref.extractall(file.split('.')[0])
    zip_ref.close()    
    

def dlfile(url, filename):
    # Open the url
    # filealready exits
    file_path = os.path.join(os.getcwd(), basename + filename)
    
    opener.retrieve(url, file_path, download_progress)
    print("\nDownload Complete\n")
    
    if filename.endswith('.zip'):
        print("Unzipping file...\n")
        unzipfile(filename)
        print("Unzipping Complete\n")



# os.path.join(os.getcwd(), basename + filename)
basename = "resources/networks/"
items = [("mars-small128.ckpt-68577", "https://doc-08-1c-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/cet8l203i1m52vsn0lt9li6mcpb2sac5/1557338400000/10781870336979197976/*/1hF6Cehn1SNZvh-M7FItSjEFojf_MVUba?e=download"), 
	("mars-small128.ckpt-68577.meta", "https://doc-08-1c-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/je87pf5eggha5m0cdgk5o8gdo8ettn5i/1557338400000/10781870336979197976/*/1FkpWjshRY1YZC3dtQT9DNUbVZLu97uqi?e=download"), 
	("mars-small128.pb", "https://doc-0g-1c-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/90hu134t8f7j8b68bt3tmha5afmghmdl/1557338400000/10781870336979197976/*/1bB66hP9voDXuoBoaCcKYY7a8IYzMMs4P?e=download"), 
    ("MOT16", "https://motchallenge.net/data/MOT16.zip")]

opener = URLopener()
pb = ProgressBar(total=100,prefix='Here', suffix='Now', decimals=3, length=50, fill='X', zfill='-')

LEFT_STR = '\n=========================================\n\t'
RGHT_STR = '\n=========================================\n'


os.system("clear")
os.system("echo '{}Downloading Requried Files....{}'".format(LEFT_STR, RGHT_STR))
for (filename, url) in items:
	dlfile(url, filename)


os.system("clear")
os.system("echo '{}Build Detection....{}'".format(LEFT_STR, RGHT_STR))
os.system('python build_npy.py')



