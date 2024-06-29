from pathlib import Path

import cv2
from rembg import remove, new_session

session = new_session(model_name='u2net')

input_path = 'C:\BANGLV\logo-represent\data\ex6.jpg'
image = cv2.imread(input_path)
# cv2.imwrite('ex6.jpg', image)

output_path = 'output.png'

with open(output_path, 'wb') as o:
    output = remove(image,
                    session=session,
                    alpha_matting=True,)
    o.write(output)