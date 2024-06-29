import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution


processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").to("cuda")


def enhance(image):
    # prepare image for the model
    inputs = processor(image, return_tensors="pt").to("cuda")

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # postprocess
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    return Image.fromarray(output)

image = Image.open('C:\BANGLV\logo-represent\ex11.png').convert('RGB')
srimage = enhance(image)
srimage.save('ex1.sr.jpg')