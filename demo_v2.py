import gradio as gr
from comfy_caller import run

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                layer2_prompt = gr.Textbox(label="Layer 2 prompting",
                                           value='portrait of a female character with long, '
                                                 'flowing hair that appears to be made of ethereal, '
                                                 'swirling patterns resembling the Northern Lights or Aurora Borealis. '
                                                 'Her face is serene, with pale skin and striking features. '
                                                 'She wears a dark-colored outfit with subtle patterns. '
                                                 'The overall style of the artwork is reminiscent of fantasy or supernatural genres',
                                    interactive=True)
            with gr.Row():
                neg_prompt = gr.Textbox(label="Negative Prompt gen Printing",
                                        value='bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, '
                                              'missing limbs, missing fingers, 3d, cgi',
                                        interactive=True)
            with gr.Row():
                cfg = gr.Slider(1,
                                         15,
                                         step=1,
                                         value=5,
                                         label="cfg",
                                         interactive=True)
                steps = gr.Slider(20,
                                60,
                                step=1,
                                value=30,
                                label="steps",
                                interactive=True)
                seeds = gr.Textbox(label="Seed",
                                        value='-1',
                                        interactive=True)
            with gr.Row():
                batch = gr.Slider(2,
                                  12,
                                  step=1,
                                  value=2,
                                  label="batch",
                                  interactive=True)
                btnpng = gr.Button("Launch")
        with gr.Column():
            poutput = gr.Gallery(
                            label="Generated images", show_label=False, elem_id="gallery"
                            , columns=[1], rows=[1], object_fit="contain", height="auto")

    btnpng.click(run,
                 inputs=[layer2_prompt, neg_prompt, batch, steps, cfg, seeds],
                 outputs=[poutput])

demo.queue().launch(share=True, show_error=True, show_api=True, server_port=8989)