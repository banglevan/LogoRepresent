import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models.isnet import ISNetDIS
# from models import *

import warnings
warnings.simplefilter("ignore")

def erode_and_dilate(mask, k_size, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)

    eroded = cv2.erode(mask, kernel, iterations=iterations)
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    trimap = np.full(mask.shape, 128)
    trimap[eroded >= 254] = 255
    trimap[dilated <= 1] = 0

    return trimap

def generate_trimap(mask, threshold=0.1, iterations=2):
    threshold = threshold * 255

    trimap = mask.copy()
    trimap = trimap.astype("uint8")

    # Erode and dilate the mask
    trimap = erode_and_dilate(trimap, k_size=(5, 5), iterations=iterations)

    return trimap

if __name__ == "__main__":
    dataset_path="../demo_datasets/your_dataset"  #Your dataset path
    model_path="C:\BANGLV\logo-represent\ISNet\weights\isnet-general-use.pth"  # the model path
    result_path=""  #The folder path that you want to save the results
    input_size = [1024, 1024]
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cuda"))
    net.eval()

    im = cv2.imread('C:\BANGLV\logo-represent\ex11.png')
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    # im = im[:, :, 0]
    # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_shp = im.shape[0:2]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bicubic").type(torch.uint8)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

    if torch.cuda.is_available():
        image = image.cuda()
    result=net(image)
    result=torch.squeeze(F.upsample(result[0][0], im_shp, mode='nearest'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)
    final_result = (result*255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)[:, :, 0]
    # toc = time.time()
    # print(toc - tic)
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=150)
    # plt.suptitle(f'{toc - tic: .5f} seconds')
    ax[0].imshow(im)
    ax[1].imshow(final_result)
    plt.show()

    # final_result = final_result.astype(np.float32) / 255
    # final_result[np.where(final_result > 40)] = 255
    # final_result[np.where(final_result <= 40)] = 0
    trimap = generate_trimap(final_result)
    cv2.imwrite('trimap.png', final_result)
    # plt.imshow(final_result)
    # plt.show()
    # plt.close()
    # im_name = im_path.split('/')[-1].split('.')[0]
    # io.imsave(os.path.join(result_path,im_name+".png"), final_result)
