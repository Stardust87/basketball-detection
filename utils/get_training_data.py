import cv2
import os, glob
import numpy as np
from tqdm import tqdm

from video_to_images import convert_to_images

VIDEOS_PATH = '.\\data\\CAMERA\\extra\\videos'
OUTPUT_PATH = f'{VIDEOS_PATH}\\..\\images'
SKIPPER = -1

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

video_filenames = glob.glob(f'{VIDEOS_PATH}\\*.MP4')
frame_num = 40000

for video_name in tqdm(video_filenames):
    frame_num = convert_to_images(video_name, OUTPUT_PATH, frame_num, sampling=SKIPPER)