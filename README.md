# Basketball detection 
**Authors:** Michał Szachniewicz [(szacho)](https://github.com/szacho), Anna Klimiuk [(Stardust87)](https://github.com/Stardust87)

This project automates the process of extracting a basketball trajectory from a video. It saves results and creates a pretty visualization:
![visualization of detected trajectory](https://raw.githubusercontent.com/szacho/basketball-detection/main/assets/output_123629AA.gif)

The algorithm used for object detection is YOLO, forked from https://github.com/ultralytics/yolov5. It has been trained on a custom dataset of privately recorded videos. The whole process is automated and it can handle a directory of videos one by one. 

## Evaluation of the model
The test set was created from two sceneries. Scenery *0* was present in a training set, but different recordings were used. Scenery *1* was completely new to the model. Every video was split into 4 categories:
- *in_hands* -- the ball is in or near the hands of a player
- *in_air* -- the ball was thrown and is in the air until it hits something
- *near_basket* -- the ball is close to the basket or even going through one (it can be occluded by a rim etc.)
- *bouncing* -- the ball is bouncing on the ground 

Furthermore, an additional category *custom* was added and it consisted of different images from the internet to assure that the model can generalize well.  We evaluated the performance of the model on every scenery and category by intersection over union and distance error between $(x, y)$ and $(\hat{x}, \hat{y})$ in pixel space.

Full results are available in the *yolo_analysis* notebook. In short: predictions are very accurate -- the average error in the distance for unseen scenery was equal to only 1.3px! That is comparable to human error.

Below are grouped per scenery and per category boxplots of IoU and distance error on the test set. 

![boxplot grouped by scenery](https://raw.githubusercontent.com/szacho/basketball-detection/main/assets/boxplots_scenery.png)
![boxplot grouped by category](https://raw.githubusercontent.com/szacho/basketball-detection/main/assets/boxplots_category.png)
