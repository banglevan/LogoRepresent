import gradio as gr
import yaml
import numpy as np
from PIL import Image
# from pipeline_apps import PipelineApps
from isn_segment import ISNSegmentProcessor
from get_built_in_styles import styles as FooocusStyles

with open(r"C:\BANGLV\logo-represent\configs\fooocus_api_configs.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

styles = list(FooocusStyles.keys())#cfg['fooocus']['style']
models = cfg['fooocus']['base_model_name']
aspects = cfg['fooocus']['aspect_ratios']
loras = cfg['fooocus']['loras']
perfs = cfg['fooocus']['perfs']
# pipeline = PipelineApps()
segmenter = ISNSegmentProcessor()

with gr.Blocks() as demo:
    with gr.Tab('PNG Creator Only'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    oimage = gr.ImageEditor(
                        type="numpy",
                        crop_size="1:1", image_mode='RGB'
                    )
                    # oimage = gr.Image(label='Output Image')
                with gr.Row():
                    btnpng = gr.Button("Launch for PNG")
            with gr.Column():
                poutput = gr.Image(label='Output Image')
    btnpng.click(segmenter.inference_on_image, inputs=[oimage], outputs=[poutput])
    with gr.Tab("Generation"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image = gr.ImageEditor(
                                            type="numpy",
                                            crop_size="1:1",
                                            )
                with gr.Row():
                    nimages = gr.Slider(1,
                                        24,
                                        step=1,
                                        value=1,
                                        label="target images number",
                                        info='Number of generated images',
                                        interactive=True)
                with gr.Row():
                    canny_stop = gr.Slider(0.1,
                                           1.0,
                                           step=0.05,
                                           value=0.85,
                                           label="control stop at",
                                           interactive=True)
                    canny_weight = gr.Slider(0.1,
                                             2.0,
                                             step=0.05,
                                             value=1.0,
                                             label="control length",
                                             interactive=True)

                with gr.Row():
                    style_image = gr.Image(label='Stylist Image')
                    with gr.Column():
                        style_stop = gr.Slider(0.1,
                                                 1.0,
                                                 step=0.05,
                                                 value=0.85,
                                                 label="control stop at",
                                                 interactive=True)

                        style_weight = gr.Slider(0.1,
                                         2.0,
                                         step=0.05,
                                         value=1.0,
                                         label="control length",
                                         interactive=True)

                        prompt = gr.Textbox(label="Prompt to Segment",
                                            info="You can input anything to segment",
                                            value="logo, white background, simple background",
                                            interactive=True)

                        neg_prompt = gr.Textbox(label="Prompt to Segment",
                                                info="You can input anything to segment",
                                                value="shadow",
                                                interactive=True)

                with gr.Row():
                    model = gr.Dropdown(
                        models, label="base model",
                        info="base stable diffusion model",
                        value=models[0], interactive=True
                    )
                    style = gr.Dropdown(
                        styles, label="built-in style",
                        info="embbeded style",
                        value=styles[0], interactive=True
                    )
                    aspect = gr.Dropdown(
                        aspects, label="output resolution",
                        info="output image size",
                        value="1024*1024", interactive=True
                    )
                    lora = gr.Dropdown(
                        loras, label="performed lora",
                        info="additional lora",
                        value=loras[0], interactive=True
                    )

            with gr.Column():
                with gr.Row():
                    gallery = gr.Gallery(
                        label="Generated images", show_label=False, elem_id="gallery"
                        , columns=[1], rows=[1], object_fit="contain", height="auto")
                with gr.Row():
                    perf = gr.Dropdown(
                        perfs,
                        label="running performance",
                        info="Hyper-SD as fastest option",
                        value='Hyper-SD', interactive=True
                    )
                    btn = gr.Button("Launch")
                    btnsm = gr.Button("PNG Generator")

                with gr.Row():
                    png_images = gr.Gallery(
                        label="PNG-4channels images", show_label=False, elem_id="gallery"
                        , columns=[1], rows=[1], object_fit="contain", height="auto")

                with gr.Row():
                    with gr.Row():
                        gr.Markdown("""
                        # Usage Instruction
                        0. Set the needed to generate images number at `target images number`
                        1. Select a garment image
                        2. Crop the printing region by manually, hit `Enter` when completed selestion
                        3. Sliding the `Control stop at` and `Control weight`
                            `Control stop at` commonly set as 0.8~1.0
                            higher `Control weight` higher similar with the original input image
                        4. Select the style driven image (second image layout in the left)
                        5. Sliding the `Control stop at` and `Control weight`
                            Same as step (3)
                        6. Any choices for other options of `performance`, `lora`, `base model`
                        7. Click `Launch` to generate images
                        """)

        # btn.click(pipeline.image_variation,
        #           inputs=[prompt, neg_prompt, perf,
        #                 aspect, nimages, style,
        #                 lora, image, style_image,
        #                 canny_stop, canny_weight,
        #                 style_stop, style_weight],
        #           outputs=[gallery])
        # btnsm.click(segmenter.inference_on_batch, inputs=[gallery], outputs=[png_images])

demo.launch(share=False, show_error=True, show_api=True, server_port=8989)