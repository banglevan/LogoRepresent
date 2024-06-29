import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm
from vit_matting import VitMatting
import torch, gc
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from ISNet.models.isnet import ISNetDIS
from typing import List
import yaml
import warnings
warnings.simplefilter("ignore")

with open(r"C:\BANGLV\logo-represent\configs\fooocus_api_configs.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

class ISNSegmentProcessor():
    def __init__(self):
        self.configurations()
        self.model = self.load_model()
        self.matter = VitMatting()

    def configurations(self):
        self.input_size = [1024, 1024]
        self.model_path = cfg['saliency_segment']['model_path']

    def load_model(self):
        net = ISNetDIS()
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(self.model_path))
            net = net.cuda()
        else:
            net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        net.eval()
        return net

    def pre_process(self, images: List[np.ndarray]):
        assert len(images) > 0
        self.im_shp = images[0].shape[0:2]
        images = np.array(images)
        im_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        im_tensor = F.upsample(im_tensor, self.input_size, mode="bilinear").type(torch.uint8)
        infer_tensors = torch.divide(im_tensor, 255.0)
        infer_tensors = normalize(infer_tensors, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        if torch.cuda.is_available():
            infer_tensors = infer_tensors.to('cuda')
        return infer_tensors

    def inference(self, images):
        tensors = self.pre_process(images)
        results = self.model(tensors)
        concated = torch.cat(results[0], dim=1)
        upsampled = F.interpolate(concated, self.im_shp, mode='bicubic', align_corners=True)
        upsampled = upsampled[:, 0, :, :]
        flatten = torch.flatten(upsampled, start_dim=1)
        ma = torch.max(flatten, dim=1)[0]
        mi = torch.min(flatten, dim=1)[0]
        finals = []
        for i in range(len(images)):
            r = upsampled[i, :, :]
            fo = (r - mi[i]) / (ma[i] - mi[i])
            fr = (fo * 255).cpu().data.numpy().astype(np.uint8)
            # im = images[i].copy()
            # im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
            # im[:, :, 3] = fr

            finals.append(fr)

        return finals

    def inference_on_image(self, image):
        image = image['composite']
        images = []
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        res = self.inference(images)[0]
        matted = self.matter.run_on_image(image, res)
        return matted

    def inference_on_batch(self, gal_images):
        #todo: for demo only
        images = []
        for i in gal_images:
            path_image = i[0]
            image = cv2.imread(path_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        finals = self.inference(images)
        return finals

if __name__ == '__main__':
    processor = ISNSegmentProcessor()
    im = cv2.imread('C:\BANGLV\logo-represent\data\ex4.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    processor.inference_on_batch([im]*5)