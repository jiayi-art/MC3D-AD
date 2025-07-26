import os

import cv2
import numpy as np


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def min_max_normalize(data):
    """
    Min-Max normalize the given NumPy array.

    args:
        data (np.ndarray)

    return:
        np.ndarray: Nomalized np.ndarray。
    """
    # Calculate minimum and maximum values
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Normalize
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def save_points(points,pred,savedir):
    with open(savedir, "w") as f:
        for i,point in enumerate(points):
            f.write(f"{point[0]} {point[1]} {point[2]} {pred[i]}\n")

def visualize_compound(fileinfos, preds,masks, labels,points, cfg_vis):
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    vis_dir = cfg_vis.save_dir
    for clsname in clsnames:
        os.makedirs(os.path.join(vis_dir, clsname), exist_ok=True)
        for fileinfo, label,pred,mask,point in zip(fileinfos, labels,preds,masks,points):
            if fileinfo["clsname"] == clsname and label == 1:
                filename = fileinfo["filename"]
                filedir, filename = os.path.split(filename)
                defename = filename.split('.')[0]
                save_dir = os.path.join(vis_dir, clsname, defename+"_heatmap.txt")
                gt_save_dir = os.path.join(vis_dir, clsname, defename+"_gt.txt")
                save_points(point,min_max_normalize(pred),save_dir)
                save_points(point,mask,gt_save_dir)
                
