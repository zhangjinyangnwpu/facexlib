import argparse
import cv2
import gradio as gr
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from facexlib.matting import init_matting_model
from facexlib.utils import img2tensor


def func_matting(img):
    img = np.array(img) / 255.
    modnet = init_matting_model(device='cpu')

    # read image
    if len(img.shape) == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, 0:3]

    img_t = img2tensor(img, bgr2rgb=True, float32=True)
    normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0)

    # resize image for input
    _, _, im_h, im_w = img_t.shape
    ref_size = 512
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    img_t = F.interpolate(img_t, size=(im_rh, im_rw), mode='area')
    # inference
    _, _, matte = modnet(img_t, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    matte_show = matte * 255
    matte_show = Image.fromarray(matte_show.astype('uint8'))
    # get foreground
    matte = matte[:, :, None]
    foreground = img * matte + np.full(img.shape, 1) * (1 - matte)
    foreground_show = foreground * 255
    foreground_show = Image.fromarray(foreground_show.astype('uint8'))
    return foreground_show,matte_show


app = gr.Interface(fn=func_matting, inputs="image", outputs=["image","image"])
app.launch(share=False)
