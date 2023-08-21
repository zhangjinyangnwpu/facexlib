import torch
import gradio as gr
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_detection

def app_detection(image):
    det_net = init_detection_model("retinaface_resnet50", half=False,device='cpu')
    with torch.no_grad():
        bboxes = det_net.detect_faces(image, 0.6)
        img_draw = visualize_detection(image, bboxes)
    return img_draw

app = gr.Interface(fn=app_detection, inputs="image", outputs=["image"])
app.launch(share=True)