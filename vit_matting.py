import os

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import torch

def init_model(model, checkpoint, device):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    assert model in ['vitmatte-s', 'vitmatte-b']
    if model == 'vitmatte-s':
        config = 'ViTMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = 'ViTMatte/configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    return model


class VitMatting():
    def __init__(self):
        self.configurations()
        self.load_model()

    def configurations(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = 'ViTMatte/weights/ViTMatte_S_DIS.pth'
        self.model_type = 'vitmatte-s'

    def load_model(self):
        self.model = init_model(self.model_type, self.model_path, self.device)

    def pre_processing(self, image: np.ndarray, trimap: np.ndarray):
        image = Image.fromarray(image).convert('RGB')
        image_tensor = F.to_tensor(image).unsqueeze(0)
        trimap = Image.fromarray(trimap).convert('L')
        trimap_tensor = F.to_tensor(trimap).unsqueeze(0)

        return {
            'image': image_tensor,
            'trimap': trimap_tensor
        }

    def run_on_image(self, image: np.ndarray, trimap: np.ndarray):
        inputs = self.pre_processing(image, trimap)
        output = self.model(inputs)['phas'].flatten(0, 2)
        output = F.to_pil_image(output)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = output
        return result

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--model', type=str, default='vitmatte-s')
    parser.add_argument('--checkpoint-dir', type=str, default='weights/ViTMatte_S_DIS.pth')
    parser.add_argument('--image-dir', type=str, default=r'C:\BANGLV\logo-represent\data\ex9.jpg')
    parser.add_argument('--trimap-dir', type=str, default=r'C:\BANGLV\logo-represent\ISNet\trimap.png')
    parser.add_argument('--output-dir', type=str, default='demo/result.png')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    input = get_data(args.image_dir, args.trimap_dir)
    print('Initializing model...Please wait...')
    model = init_model(args.model, args.checkpoint_dir, args.device)
    print('Model initialized. Start inferencing...')
    alpha = infer_one_image(model, input, args.output_dir)
    print('Inferencing finished.')