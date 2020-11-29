import numpy as np

def convert_coords_matrix(boxes, width_height):
    xy1 = (boxes[:, :2]-boxes[:, 2:]/2) * width_height
    xy2 = (boxes[:, :2]+boxes[:, 2:]/2) * width_height
    return np.concatenate([xy1, xy2], axis=1)


def intersection_over_union_matrix(boxesA_yolo, boxesB_yolo, width_height_mat):
    boxesB = convert_coords_matrix(boxesB_yolo, width_height_mat)
    boxesA = convert_coords_matrix(boxesA_yolo, width_height_mat)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.max(np.stack([boxesA[:, 0], boxesB[:, 0]], axis=1), axis=1)
    yA = np.max(np.stack([boxesA[:, 1], boxesB[:, 1]], axis=1), axis=1)
    xB = np.min(np.stack([boxesA[:, 2], boxesB[:, 2]], axis=1), axis=1)
    yB = np.min(np.stack([boxesA[:, 3], boxesB[:, 3]], axis=1), axis=1)
    
    # compute the area of intersection rectangles
    x_max = np.max(np.stack([np.zeros_like(xA), xB - xA + 1], axis=1), axis=1)
    y_max = np.max(np.stack([np.zeros_like(xA), yB - yA + 1], axis=1), axis=1)
    interArea = x_max * y_max
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxesA[:, 2] - boxesA[:, 0] + 1) * (boxesA[:, 3] - boxesA[:, 1] + 1)
    boxBArea = (boxesB[:, 2] - boxesB[:, 0] + 1) * (boxesB[:, 3] - boxesB[:, 1] + 1)

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou