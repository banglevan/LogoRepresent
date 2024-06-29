import sys
sys.path.append("..")
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib
matplotlib.use('TkAgg')

from segment_anything import (sam_model_registry,
                          SamAutomaticMaskGenerator,
                          SamPredictor)
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import load_image
import torch

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class SAMClothesSegment():
    def __init__(self):
        self.sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.configurations()
        self.load_model()

    def configurations(self):
        self.points_per_side = 32
        self.pred_iou_thresh = 0.95
        self.stability_score_thresh = 0.95
        self.crop_n_layers = 1
        self.crop_n_points_downscale_factor = 2
        self.min_mask_region_area = 100  # Requires open-cv to run post-processing

    def load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self.mask_generator = SamAutomaticMaskGenerator(
                                model=sam,
                                points_per_side=self.points_per_side,
                                pred_iou_thresh=self.pred_iou_thresh,
                                stability_score_thresh=self.stability_score_thresh,
                                crop_n_layers=self.crop_n_layers,
                                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                                min_mask_region_area=self.min_mask_region_area,  # Requires open-cv to run post-processing
                                                        )

    def _pre_process(self, image):
        pass

    def _post_process(self, image):
        pass

    def inference(self, image):
        image = load_image(image, itype='np')
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    image = cv2.imread('C:\BANGLV\logo-represent\data\ex3.png')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    segmenter = SAMClothesSegment()
    segmenter.inference(image)