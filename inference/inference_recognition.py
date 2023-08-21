import argparse
import glob
import math
import numpy as np
import os
import torch
import cv2
from facexlib.recognition import init_recognition_model
# from facexlib.recognition import ResNetArcFace, cosin_metric, load_image

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (256, 256))
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    image = torch.from_numpy(image)
    print(image.shape)
    return image

if __name__ == '__main__':
    # python3 inference/inference_recognition.py --folder1 assets/facevertify/diff_person/auth --folder2 assets/facevertify/diff_person/train
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder1', type=str)
    parser.add_argument('--folder2', type=str)
    parser.add_argument('--device', type=str,default='cpu')
    parser.add_argument('--model_path', type=str, default='/Users/zhangjinyang/opt/anaconda3/lib/python3.9/site-packages/facexlib/weights/recognition_arcface_ir_se50.pth')

    args = parser.parse_args()

    img_list1 = sorted(glob.glob(os.path.join(args.folder1, '*')))
    img_list2 = sorted(glob.glob(os.path.join(args.folder2, '*')))
    print(img_list1, img_list2)
    model = init_recognition_model(model_name='arcface',device=args.device)
    model.load_state_dict(torch.load(args.model_path,map_location=args.device))
    model.to(args.device)
    model.eval()

    dist_list = []
    identical_count = 0
    for idx, (img_path1, img_path2) in enumerate(zip(img_list1, img_list2)):
        basename = os.path.splitext(os.path.basename(img_path1))[0]
        img1 = load_image(img_path1)
        img2 = load_image(img_path2)
        
        data = torch.concatenate([img1, img2], dim=0)
        data = data.to(args.device)
        data = torch.randn(1, 3, 128, 128)
        output = model(data)
        print(output.size())
        output = output.data.cpu().numpy()
        dist = cosin_metric(output[0], output[1])
        dist = np.arccos(dist) / math.pi * 180
        print(f'{idx} - {dist} o : {basename}')
        if dist < 1:
            print(f'{basename} is almost identical to original.')
            identical_count += 1
        else:
            dist_list.append(dist)

    print(f'Result dist: {sum(dist_list) / len(dist_list):.6f}')
    print(f'identical count: {identical_count}')
