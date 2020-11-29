import json
import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import savgol_filter

def smooth_trajectory(points):
    split_trajectory = [ [points[0]] ]
    current_split = 0
    for i in range(1, len(points)):
        distance = np.linalg.norm(np.array(points[i - 1])-np.array(points[i]))
        if distance < 15:
            split_trajectory[current_split].append(points[i])
        else:
            split_trajectory.append([ points[i] ])
            current_split += 1
    
    new_X, new_Y = [], []

    for split in split_trajectory:
        X = np.array([ p[0] for p in split ])
        Y = np.array([ p[1] for p in split ])

        if len(Y) > 20:
            y_hat = savgol_filter(Y, 15, 3)
            Y = [ int(round(y)) for y in y_hat ]

        new_X.extend(X)
        new_Y.extend(Y)

    return list(zip(new_X, new_Y))

def get_pixel_coords(obj, img_height, img_width):
    coords = obj['relative_coordinates']
    center_x = int(img_width*coords['center_x'])
    center_y = int(img_height*coords['center_y'])
    obj_width = int(img_width*coords['width'])
    obj_height = int(img_height*coords['height'])
    
    return center_x, center_y, obj_width, obj_height

def draw_circle(image, x, y, w, h, conf):
    r = max(int(w/2), int(h/2))
    cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    cv2.circle(image, (x, y), 2, (0, 255, 0), -2)

def draw_box(image, x, y, w, h):
    cv2.rectangle(image,(int(x-w/2), int(y+h/2)), (int(x+w/2), int(y-h/2)), (255,0,255), 2)

def draw_line_fragment(image, points, color=(135, 0, 190), thickness=2, max_distance=30):
    for i in range(1, len(points)):
        distance = np.linalg.norm(np.array(points[i - 1])-np.array(points[i]))
        if distance < max_distance:
            image = cv2.line(image, points[i - 1], points[i], color, thickness)

    return image

def draw_glowing_line(img, points, max_distance):
    line_img = np.zeros_like(img)
    line_img = draw_line_fragment(line_img, points, color=(135, 0, 190), thickness=2, max_distance=max_distance)
    line_img = draw_line_fragment(line_img, points, color=(0, 0, 255), thickness=1, max_distance=max_distance)

    blurred_line = cv2.GaussianBlur(line_img, (5,5), 0)

    img_gray = cv2.cvtColor(blurred_line, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
    mask = np.stack([mask, mask, mask], axis=2)
    mask = np.where(mask > 0, 0.6, 1)
    img = img*mask

    return cv2.addWeighted(img.astype('uint8'), 1, blurred_line, 1.6, 0)


def draw_streak(image, points, max_distance):
    for i in range(1, len(points)):
        thickness = 2
        distance = np.linalg.norm(np.array(points[i - 1])-np.array(points[i]))
        if distance < max_distance:
            image = cv2.line(image, points[i - 1], points[i], (255, 0, 255), thickness)

    return image

def process_image(img_id, res, points, images_path, ball_conf, max_distance, style, tail, smooth):
    assert style in ['glowing', 'fast'], f"style should be one of {['glowing', 'fast']}, instead of {style}"

    filename = images_path+res[img_id]['filename'].split('/')[-1]
    objects = res[img_id]['objects'] 

    image = cv2.imread(filename,1)
    height, width, _ = image.shape

    obj_to_draw = None
    max_conf = 0
    for obj in objects:
        new_point = None
        if obj['class_id'] == 0 and obj['confidence'] >= ball_conf and obj['confidence'] > max_conf:
            obj_to_draw = obj
            max_conf = obj['confidence']

    if obj_to_draw: 
        ball_x, ball_y, obj_width, obj_height = get_pixel_coords(obj_to_draw, height, width)
        new_point = ball_x, ball_y
        points.append(new_point)
        draw_circle(image, ball_x, ball_y, obj_width, obj_height, obj_to_draw['confidence'])
    
    points = list(filter(lambda pt: pt is not None, points))
    if len(points) > 30 and smooth:
        points = smooth_trajectory(points)

    if tail > 0:
        points = points[-tail:]

    if style == 'fast':
        image = draw_streak(image, points, max_distance)
    elif style == 'glowing':
        image = draw_glowing_line(image, points, max_distance)

    return image, points

def draw_trajectories(input_file,images_path,output_path, ball_conf=0.3, max_distance=30, style='glowing', tail=False, smooth=True):
    with open(input_file) as f:
        res = json.load(f)

    points = []
    progress_bar = tqdm(range(len(res))) 
    for img_id in progress_bar:
        progress_bar.set_description("Drawing trajectories")
        image, points = process_image(img_id, res, points, images_path, ball_conf, max_distance, style, tail, smooth)
        cv2.imwrite(output_path+f'imgp{img_id:04d}.jpg', image)

if __name__ == "__main__":
    VIDEO_NAME = 'michal1'
    INPUT_FILE = f'data/{VIDEO_NAME}/result.json'
    IMAGES_PATH = f'data/{VIDEO_NAME}/images_raw/'
    OUTPUT_PATH = f'data/{VIDEO_NAME}/images_draw/'
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    BALL_CONF = 0.3

    process_file(INPUT_FILE,IMAGES_PATH,OUTPUT_PATH)
