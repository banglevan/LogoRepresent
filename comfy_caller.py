import json
from urllib import request, parse
import random
from PIL import Image
import io
from matplotlib import pyplot as plt
import random
#This is the ComfyUI api prompt format.
#If you want it for a specific workflow you can "enable dev mode options"
#in the settings of the UI (gear beside the "Queue Size: ") this will enable
#a button on the UI to save workflows in api format.
#keep in mind ComfyUI is pre alpha software so this format will change a bit.
#this is the one for the default workflow
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)

import uuid
import json
import urllib.request
import urllib.parse

server_address = "192.168.0.44:9999"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    res = json.loads(urllib.request.urlopen(req).read())
    print(res)
    return res

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images

prompt_text_ = """    
                {
                  "11": {
                            "inputs": {
                              "clip_name1": "clip_g.safetensors",
                              "clip_name2": "clip_l.safetensors",
                              "clip_name3": "t5xxl_fp16_e4m3fn.safetensors"
                            },
                            "class_type": "TripleCLIPLoader",
                            "_meta": {
                              "title": "TripleCLIPLoader"
                            }
                          },
                  "13": {
                            "inputs": {
                              "shift": 3,
                              "model": [
                                "252",
                                0
                              ]
                            },
                            "class_type": "ModelSamplingSD3",
                            "_meta": {
                              "title": "ModelSamplingSD3"
                            }
                          },
                  "67": {
                            "inputs": {
                              "conditioning": [
                                "71",
                                0
                              ]
                            },
                            "class_type": "ConditioningZeroOut",
                            "_meta": {
                              "title": "ConditioningZeroOut"
                            }
                          },
                  "68": {
                            "inputs": {
                              "start": 0.1,
                              "end": 1,
                              "conditioning": [
                                "67",
                                0
                              ]
                            },
                            "class_type": "ConditioningSetTimestepRange",
                            "_meta": {
                              "title": "ConditioningSetTimestepRange"
                            }
                          },
                  "69": {
                            "inputs": {
                              "conditioning_1": [
                                "68",
                                0
                              ],
                              "conditioning_2": [
                                "70",
                                0
                              ]
                            },
                            "class_type": "ConditioningCombine",
                            "_meta": {
                              "title": "Conditioning (Combine)"
                            }
                          },
                  "70": {
                            "inputs": {
                              "start": 0,
                              "end": 0.1,
                              "conditioning": [
                                "71",
                                0
                              ]
                            },
                            "class_type": "ConditioningSetTimestepRange",
                            "_meta": {
                              "title": "ConditioningSetTimestepRange"
                            }
                          },
                  "71": {
                            "inputs": {
                              "text": "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi",
                              "clip": [
                                "252",
                                1
                              ]
                            },
                            "class_type": "CLIPTextEncode",
                            "_meta": {
                              "title": "CLIP Text Encode (Negative Prompt)"
                            }
                          },
                  "135": {
                            "inputs": {
                              "width": 1024,
                              "height": 1024,
                              "batch_size": 6
                            },
                            "class_type": "EmptySD3LatentImage",
                            "_meta": {
                              "title": "EmptySD3LatentImage"
                            }
                          },
                  "231": {
                            "inputs": {
                              "samples": [
                                "271",
                                0
                              ],
                              "vae": [
                                "252",
                                2
                              ]
                            },
                            "class_type": "VAEDecode",
                            "_meta": {
                              "title": "VAE Decode"
                            }
                          },
                  "233": {
                            "inputs": {
                              "images": [
                                "231",
                                0
                              ]
                            },
                            "class_type": "PreviewImage",
                            "_meta": {
                              "title": "Preview Image"
                            }
                          },
                  "252": {
                            "inputs": {
                              "ckpt_name": "sd3_medium_incl_clips_t5xxlfp16.safetensors"
                            },
                            "class_type": "CheckpointLoaderSimple",
                            "_meta": {
                              "title": "Load Checkpoint"
                            }
                          },
                  "271": {
                            "inputs": {
                              "seed": 124346677,
                              "steps": 28,
                              "cfg": 4.5,
                              "sampler_name": "dpmpp_2m",
                              "scheduler": "sgm_uniform",
                              "denoise": 1,
                              "model": [
                                "13",
                                0
                              ],
                              "positive": [
                                "273",
                                0
                              ],
                              "negative": [
                                "69",
                                0
                              ],
                              "latent_image": [
                                "135",
                                0
                              ]
                            },
                                "class_type": "KSampler",
                                "_meta": {
                                  "title": "KSampler"
                                }
                              },
                  "273": {
                            "inputs": {
                              "clip_l": "the background is dominated by deep red and purples, creating a mysterious and dramatic atmosphere similar to a volcanic explosion",
                              "clip_g": "the background is dominated by deep red and purples, creating a mysterious and dramatic atmosphere similar to a volcanic explosion",
                              "t5xxl": "portrait of a female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. Her face is serene, with pale skin and striking features. She wears a dark-colored outfit with subtle patterns. The overall style of the artwork is reminiscent of fantasy or supernatural genres\n",
                              "empty_padding": "none",
                              "clip": [
                                "252",
                                1
                              ]
                            },
                            "class_type": "CLIPTextEncodeSD3",
                            "_meta": {
                              "title": "CLIPTextEncodeSD3"
                            }
                          }
                        }
            """

prompt_text =    """{
                  "6": {
                        "inputs": {
                                      "text": "a photo of t shirt printing, logo 'dad is my friend', dad and son in brown",
                                      "clip": [
                                        "252",
                                        1
                                      ]
                                    },
                        "class_type": "CLIPTextEncode",
                        "_meta": {
                          "title": "CLIP Text Encode (Prompt)"
                                }
                        },
                  "11": {
                        "inputs": {
                                      "clip_name1": "clip_g.safetensors",
                                      "clip_name2": "clip_l.safetensors",
                                      "clip_name3": "t5xxl_fp8_e4m3fn.safetensors"
                                    },
                        "class_type": "TripleCLIPLoader",
                        "_meta": {
                          "title": "TripleCLIPLoader"
                                }
                        },
                  "13": {
                        "inputs": {
                                      "shift": 3,
                                      "model": [
                                        "252",
                                        0
                                      ]
                                    },
                        "class_type": "ModelSamplingSD3",
                        "_meta": {
                          "title": "ModelSamplingSD3"
                                }
                        },
                  "67": {
                        "inputs": {
                                  "conditioning": [
                                    "71",
                                    0
                                  ]
                                },
                        "class_type": "ConditioningZeroOut",
                        "_meta": {
                          "title": "ConditioningZeroOut"
                                }
                        },
                  "68": {
                        "inputs": {
                          "start": 0.1,
                          "end": 1,
                          "conditioning": [
                            "67",
                            0
                          ]
                        },
                        "class_type": "ConditioningSetTimestepRange",
                        "_meta": {
                          "title": "ConditioningSetTimestepRange"
                        }
                      },
                  "69": {
                        "inputs": {
                          "conditioning_1": [
                            "68",
                            0
                          ],
                          "conditioning_2": [
                            "70",
                            0
                          ]
                        },
                        "class_type": "ConditioningCombine",
                        "_meta": {
                          "title": "Conditioning (Combine)"
                        }
                      },
                  "70": {
                        "inputs": {
                          "start": 0,
                          "end": 0.1,
                          "conditioning": [
                            "71",
                            0
                          ]
                        },
                        "class_type": "ConditioningSetTimestepRange",
                        "_meta": {
                          "title": "ConditioningSetTimestepRange"
                        }
                      },
                  "71": {
                        "inputs": {
                          "text": "bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi",
                          "clip": [
                            "252",
                            1
                          ]
                        },
                        "class_type": "CLIPTextEncode",
                        "_meta": {
                          "title": "CLIP Text Encode (Negative Prompt)"
                        }
                      },
                  "135": {
                        "inputs": {
                          "width": 1024,
                          "height": 1024,
                          "batch_size": 8
                        },
                        "class_type": "EmptySD3LatentImage",
                        "_meta": {
                          "title": "EmptySD3LatentImage"
                        }
                      },
                  "231": {
                        "inputs": {
                          "samples": [
                            "271",
                            0
                          ],
                          "vae": [
                            "252",
                            2
                          ]
                        },
                        "class_type": "VAEDecode",
                        "_meta": {
                          "title": "VAE Decode"
                        }
                      },
                  "233": {
                        "inputs": {
                          "images": [
                            "231",
                            0
                          ]
                        },
                        "class_type": "PreviewImage",
                        "_meta": {
                          "title": "Preview Image"
                        }
                      },
                  "252": {
                        "inputs": {
                          "ckpt_name": "sd3_medium_incl_clips_t5xxlfp16.safetensors"
                        },
                        "class_type": "CheckpointLoaderSimple",
                        "_meta": {
                          "title": "Load Checkpoint"
                        }
                      },
                  "271": {
                        "inputs": {
                          "seed": 945512652412924,
                          "steps": 28,
                          "cfg": 4.5,
                          "sampler_name": "dpmpp_2m",
                          "scheduler": "sgm_uniform",
                          "denoise": 1,
                          "model": [
                            "13",
                            0
                          ],
                          "positive": [
                            "6",
                            0
                          ],
                          "negative": [
                            "69",
                            0
                          ],
                          "latent_image": [
                            "135",
                            0
                          ]
                        },
                        "class_type": "KSampler",
                        "_meta": {
                          "title": "KSampler"
                        }
                    }
                }
                """


ws = websocket.WebSocket()
prompt = json.loads(prompt_text, strict=False)
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

def run(posprompt, negprompt, batch, steps, cfg, seed):
    if seed == '-1':
        seed_ = random.randint(1000000000, 1000000000000000)
    else:
        try:
            seed_ = int(seed)
        except:
            seed_ = random.randint(1000000000, 1000000000000000)
    prompt["271"]["inputs"]["seed"] = seed_
    prompt["271"]["inputs"]["steps"] = steps
    prompt["271"]["inputs"]["cfg"] = cfg

    prompt["6"]["inputs"]["text"] = posprompt
    prompt["71"]["inputs"]["text"] = negprompt
    prompt["135"]["inputs"]["batch_size"] = batch
    images = get_images(ws, prompt)['233']
    arrays = [Image.open(io.BytesIO(images[i])) for i in range(len(images))]
    return arrays