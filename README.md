# Basketball detection 
**Authors:** Michał Szachniewicz [(szacho)](https://github.com/szacho), Anna Klimiuk [(Stardust87)](https://github.com/Stardust87)

This project automates the process of extracting a basketball trajectory from a video. It saves results and creates a pretty visualization:

![visualization of detected trajectory](https://raw.githubusercontent.com/szacho/basketball-detection/main/assets/output_123629AA.gif)
![visualization of detected trajectory 2](https://raw.githubusercontent.com/szacho/basketball-detection/main/assets/output_test_163955AA.gif)

The algorithm used for object detection is YOLO, forked from https://github.com/ultralytics/yolov5. It has been trained on a custom dataset of privately recorded videos. 

## Usage
Clone this repo and pull YOLOv5 code:
```bash
git clone https://github.com/Stardust87/basketball-detection
cd basketball-detection
git submodule update --init
```

Install dependencies:
```
pip install -r requirements.txt
```

Process a video:
```
python video_pipeline.py [VIDEO_PATH] --fps 30 --clean
```

This pipeline will:
- extract frames from a video
- detect basketballs on each frame using YOLOv5 model
- extract a smoothed trajectory from the detected bounding boxes and draw it
- save the results in a folder

See `video_pipeline.py` for details.
