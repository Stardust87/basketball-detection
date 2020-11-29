import pandas as pd
import numpy as np
import json
from pathlib import Path
import cv2

images_path = Path('./data/datasets/test/images')

labels_scheme = {
    'image_id': [],
    'label': [],
    'center_x': [],
    'center_y': [],
    'width': [],
    'height': [],
    'conf': [],
    'category': [],
    'scenery': [],
    'img_width': [],
    'img_height': []
}

def read_labels(dir_path, category=np.nan, labels_scheme=labels_scheme):
    txt_files = dir_path.glob('*.txt')


    for txt in txt_files:
        if category == 'custom':
            img = cv2.imread(str(images_path/category/(txt.stem+'.jpg')))
            shape = img.shape
        else:
            shape = (720, 1280, 3)

        labels_scheme['img_height'].append(shape[0])
        labels_scheme['img_width'].append(shape[1])

        with open(txt) as f:
            lines = [l.split() for l in f.readlines()]
            if len(lines) == 1:
                data = lines[0]
            else:
                lines.sort(key=lambda l: float(l[1]), reverse=True)
                data = lines[0]

            if len(data) == 6:
                labels_scheme['conf'].append(float(data[1]))
                shift = 1
            else: 
                labels_scheme['conf'].append(np.nan)
                shift = 0

            labels_scheme['image_id'].append(txt.stem+'_'+category)
            labels_scheme['label'].append(int(data[0]))
            labels_scheme['center_x'].append(float(data[1+shift]))
            labels_scheme['center_y'].append(float(data[2+shift]))
            labels_scheme['width'].append(float(data[3+shift]))
            labels_scheme['height'].append(float(data[4+shift]))
            labels_scheme['category'].append(category)
            
            if category in ['in_air', 'near_basket', 'in_hands', 'bouncing']:
                video_id = int(txt.stem.split('_')[0][1:])
                if video_id in [1, 2, 3, 7, 8]:
                    scenery = 1
                else:
                    scenery = 0
            else:
                scenery = np.nan
                
            labels_scheme['scenery'].append(scenery)

    # labels_df = pd.DataFrame.from_dict(labels_scheme)
    return labels_scheme


if __name__ == "__main__":
    categories = ['in_air', 'near_basket', 'in_hands', 'bouncing', 'custom']
    category = categories[4]
    for category in categories:
        # PATH = Path(f'./data/datasets/test/labels/{category}_txt')
        PATH = Path(f'./data/datasets/test/inference/{category}')
        labels_scheme = read_labels(PATH, category=category, labels_scheme=labels_scheme)
        
    df = pd.DataFrame.from_dict(labels_scheme)
    print(df)
    # df.to_csv(f'./data/datasets/test/inference/ground_truth.csv', index=False)
    df.to_csv(f'./data/datasets/test/inference/predictions.csv', index=False)