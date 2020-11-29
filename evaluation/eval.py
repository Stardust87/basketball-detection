import os
from pathlib import Path

category = 'bouncing'
ROOT_PATH = Path('./data/datasets/test/images/')
output_path = ROOT_PATH/'..'/'inference'/category

MODEL_WEIGHTS = 'models/yolov5s_epochs_200_lr_0.0001-0.002_bs_32.pt'
CONF_THRESHOLD = 0.5

def predict(images_path, output_path):
    cmd = f'python3 yolov5/detect.py --weights {MODEL_WEIGHTS} --conf {CONF_THRESHOLD} --source {images_path} --save-dir {output_path} --save-txt --save-img --save-conf'
    os.system(cmd)

if __name__ == "__main__":
    predict(ROOT_PATH/category, output_path)