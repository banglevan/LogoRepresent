import gradio as gr
import time


def sleep(im):
    time.sleep(5)
    return [im["background"], im["layers"][0], im["layers"][1], im["composite"]]


def predict(im):
    return im["composite"]

def ttt(im):
    return str(im["composite"].shape)

with gr.Blocks() as demo:
    with gr.Row():
        im = gr.ImageEditor(
            type="numpy",
            crop_size="1:1",
        )
        im_preview = gr.Image()
        txt = gr.Textbox(label="")
        btn = gr.Button("Preview")
    im.change(predict, outputs=im_preview, inputs=im, show_progress="hidden")
    btn.click(ttt, inputs=im, outputs=txt, show_progress="")

if __name__ == "__main__":
    demo.launch()