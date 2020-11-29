from utils import convert_to_video, convert_to_images, draw_trajectories
import os, json
import glob
import shutil

def process_video(path, cfg):
    STAGES = cfg['stages']

    video_dir = path.split('.')[0]
    video_name = video_dir.split('/')[-1]
    images_raw_path = f'{video_dir}/images_raw'
    images_draw_path = f'{video_dir}/images_draw'
    if not os.path.exists(video_dir): os.makedirs(video_dir)
    if not os.path.exists(images_raw_path): os.makedirs(images_raw_path)
    if not os.path.exists(images_draw_path): os.makedirs(images_draw_path)
    
    # copy video file
    video_path = f"{video_dir}/{path.split('/')[-1]}"
    shutil.copyfile(path, video_path)

    # convert video to images
    det_cfg = cfg['detection']
    if 0 in STAGES:
        convert_to_images(video_path, images_raw_path, sampling=det_cfg['sampling'])
    
    # detect balls using YOLO
    if 1 in STAGES:
        detect_cmd = f"python3 yolov5/detect.py --weights models/{det_cfg['model_weights']} --conf 0.1 --source {images_raw_path}/ --img-size {det_cfg['image_size']} --save-img --augment"
        os.system(detect_cmd)

    # draw trajectories
    if 2 in STAGES:
        draw_cfg = cfg['drawing']
        draw_trajectories(f'{video_dir}/results.json', images_raw_path+'/', images_draw_path+'/', 
                        ball_conf=draw_cfg['min_confidence'], max_distance=draw_cfg['max_distance'],
                        style=draw_cfg['style'], tail=draw_cfg['tail'], smooth=draw_cfg['smoothing'])

    # make video
    if 3 in STAGES:
        convert_to_video(images_draw_path, f"{video_dir}/output_{video_name}.avi", fps=cfg['output']['fps'])
    
    if cfg['output']['clean_data']:
        os.remove(path)

if __name__ == "__main__":
    # PATH = 'processed_data/163342AA.MP4'
    with open('configs/precise.json') as f:
        cfg = json.load(f)

    VIDEOS_PATH = cfg['videos_path']
    filenames = glob.glob(f'{VIDEOS_PATH}*.MP4')
    
    for idx, video_filename in enumerate(filenames):
        video_filename = video_filename.replace('\\', '/')
        print(f"VIDEO {idx+1}/{len(filenames)}\n")
        process_video(video_filename, cfg)

