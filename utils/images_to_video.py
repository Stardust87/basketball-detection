import cv2
import numpy as np
import glob
from tqdm import tqdm



def convert_to_video(images_path, output_path, fps=240):
    img_array = []
    filenames = list(glob.glob(f'{images_path}/*.jpg'))
    filenames.sort(key=lambda x: int(x.split('/')[-1].split('.')[0][4:]))
    progress_bar = tqdm(filenames)
    for filename in progress_bar:
        progress_bar.set_description("Making a video")
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":

    VIDEO_NAME = 'michal1'
    OUTPUT_PATH = f'data/{VIDEO_NAME}/'
    convert_to_video(VIDEO_NAME,OUTPUT_PATH )
