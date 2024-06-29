from PIL import Image
import numpy as np
import cv2

def load_image(image, itype='pil'):
    if type(image) is str:
        if itype == 'pil':
            image = Image.open(image)
            return image
        else:
            image = cv2.imread(image)
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif type(image) is np.ndarray:
        if itype == 'pil':
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return image
        else:
            return image
    else:
        raise TypeError(type(image))