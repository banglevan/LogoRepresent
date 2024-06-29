import cv2
import numpy as np
from typing import List, Any
import json
import os
import base64
import requests
import yaml
from PIL import Image
import PIL
from io import BytesIO

with open(r"C:\BANGLV\logo-represent\configs\fooocus_api_configs.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def read_image(image_name: str) -> str:
    """
    Read image from file
    Args:
        image_name (str): Image file name
    Returns:
        str: Image base64
    """
    path = image_name#os.path.join('imgs', image_name)
    with open(path, "rb") as f:
        image = f.read()
        f.close()
    return base64.b64encode(image).decode('utf-8')

def image_to_base64(image: np.ndarray) -> str:
    pimage = PIL.Image.fromarray(image).convert('RGB')
    im_file = BytesIO()
    pimage.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return im_b64

def upscale_vary(params: dict) -> dict:
    """
    Upscale or Vary
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.img_upscale}",
        data=data,
        timeout=300)
    return response.json()

def inpaint_outpaint(params: dict = None) -> dict:
    """
    Inpaint or Outpaint
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.inpaint_outpaint}",
        data=data,
        timeout=300)
    return response.json()


def image_prompt(params: dict) -> dict:
    """
    Image Prompt
    Args:
        params (dict): Params
    Returns:
        dict: Response
    """
    data = json.dumps(params)
    response = requests.post(
        url=f"{Config.fooocus_host}{Config.img_prompt}",
        data=data,
        timeout=300)
    return response.json()

class PipelineApps:
    def __init__(self):
        self.configurations()

    def configurations(self):
        self.fooocus_host = cfg['network']['host']
        self.fooocus_port = cfg['network']['port']
        self.api_text2img_ip = cfg['api']['text2img_ip']
        self.api_img_upscale = cfg['api']['img_upscale']
        self.api_inpaint_outpaint = cfg['api']['inpaint_outpaint']
        self.api_img_prompt = cfg['api']['img_prompt']

        self.mock_image1 = 'C:\BANGLV\logo-represent\data\ex3.png'
        self.mock_image2 = 'C:\BANGLV\logo-represent\data\ex4.png'
        self.default_prompt = 'logo, white background, simple backgrround'
        self.default_neg_prompt = 'shadow'
        self.mock_string1 = read_image(self.mock_image1)
        self.mock_string2 = read_image(self.mock_image2)
        self.inumber = 2
        self.canny_stop = 0.8
        self.canny_weight = 0.8
        self.iprompt_stop = 0.8
        self.iprompt_weight = 0.8
        self.performance_type = 'Speed'
        self.default_aspect = '1152*896'
        self.default_style = 'Fooocus V2'
        self.default_lora = "sd_xl_offset_example-lora_1.0.safetensors"
        self.timeout = 300

    def text2image_image_prompt(self, params: dict) -> dict:
        """
        Text to image with image Prompt
        Args:
            params (dict): Params
        Returns:
            dict: Response
        """
        params["outpaint_selections"] = ["Left", "Right"]
        data = json.dumps(params)
        response = requests.post(
            url=f"{self.fooocus_host}:{self.fooocus_port}{self.api_text2img_ip}",
            data=data,
            timeout=self.timeout)
        return response.json()

    def set_api_params(self, image_prompt: dict):
        t2i_ip_params = {
            "prompt": self.default_prompt,
            "negative_prompt": self.default_neg_prompt,
            "performance_selection": self.performance_type,
            "aspect_ratios_selection": self.default_aspect,
            "image_number": self.inumber,
            "sharpness": 7,
            "style_selections": [
                                self.default_style,
                                ],
            "loras": [
                {
                    "enabled": True,
                    "model_name": self.default_lora,
                    "weight": 0.1
                },
                {
                    "enabled": True,
                    "model_name": "None",
                    "weight": 1
                },
                {
                    "enabled": True,
                    "model_name": "None",
                    "weight": 1
                },
                {
                    "enabled": True,
                    "model_name": "None",
                    "weight": 1
                },
                {
                    "enabled": True,
                    "model_name": "None",
                    "weight": 1
                }
            ],
            "image_prompts": image_prompt['image_prompts'],
            "async_process": False
                        }
        return t2i_ip_params

    def image_variation(self, text_prompt: str, text_neg_prompt: str,
                        performance_type: str, aspects: str,
                        inumber: int, style: str, lora: str,
                        iprompt1: dict, iprompt2: Any,
                        canny_stop: float, canny_weight: float,
                        iptpm_stop: float, iptpm_weight: float) -> List[np.ndarray]:
        """
        call Fooocus-API to generate an image variation
        from images as the driven edge and style
        :return: image variations
        """
        self.default_prompt = text_prompt.strip()
        self.default_neg_prompt = text_neg_prompt.strip()
        self.performance_type = performance_type.strip()
        self.default_aspect = aspects.strip()
        self.inumber = inumber
        self.default_style = style.strip()
        self.default_lora = lora.strip()
        iprompt1 = iprompt1['composite']
        self.mock_string1 = image_to_base64(iprompt1)
        self.canny_stop = float(canny_stop)
        self.canny_weight = float(canny_weight)
        self.prompt_stop = float(iptpm_stop)
        self.prompt_weight = float(iptpm_weight)

        image_prompt = {}
        image_prompt['image_prompts'] = [{
                                            "cn_img": self.mock_string1,
                                            "cn_stop": self.canny_stop,
                                            "cn_weight": self.canny_weight,
                                            "cn_type": "PyraCanny"
                                        },]
        if type(iprompt2) is np.ndarray:
            self.mock_string2 = image_to_base64(iprompt2)
            image_prompt['image_prompts'].append({
                                                "cn_img": self.mock_string2,
                                                "cn_stop": self.iprompt_stop,
                                                "cn_weight": self.iprompt_weight,
                                                "cn_type": "ImagePrompt"
                                                })
        params = self.set_api_params(image_prompt)
        t2i_ip_result = self.text2image_image_prompt(params)
        images = self._response_parser(t2i_ip_result)
        return images

    def _response_parser(self, t2i_ip_result):
        nimage = len(t2i_ip_result)
        images = []
        for i in range(nimage):
            url = t2i_ip_result[i]['url']
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)
        return images

if __name__ == '__main__':
    pipeline = PipelineApps()
    params = pipeline.set_api_params()
    t2i_ip_result = pipeline.text2image_image_prompt(params)
    pipeline._response_parser(t2i_ip_result)
