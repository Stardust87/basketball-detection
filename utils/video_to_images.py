import cv2
import os
import numpy as np

def convert_to_images(video_path, output_path, current_frame=0, sampling=0):
    print("Converting the video to images...")
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)

    while capture.isOpened():
        ret, image = capture.read()
        if image is None:
            break

        if sampling > 0:
            if current_frame % (sampling + 1) == 0:
                cv2.imwrite(f'{output_path}/img{current_frame:04d}.jpg',image)    
        else:
            cv2.imwrite(f'{output_path}/img{current_frame:04d}.jpg',image)
        current_frame += 1
    
    return current_frame

if __name__ == "__main__":
    videoname = 'michal1'
    video_path = f'data/{videoname}/input.MP4'
    output_path = f'data/{videoname}/images_raw'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    convert_to_images(video_path, output_path)

