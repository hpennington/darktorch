#!/usr/bin/env python3

import time
import subprocess
import argparse

import numpy as np
import torch
import visdom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2

import darktorch
from darktorch.utils import darknet_from_cfg
from darktorch.utils import nonmax_suppression, calculate_detections
from darktorch.utils import draw_detections, draw_detections_opencv
from darktorch.utils import parse_categories, parse_data
from darktorch.utils import write_tensor, read_tensor
from darktorch.transforms import LetterboxDetect, BGR2RGB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument(
        '--weights', type=str, default='weights/yolov3.weights')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg')
    parser.add_argument('--data', type=str, default='cfg/coco.data')
    parser.add_argument('--image', type=str, default='data/dog.jpg')
    parser.add_argument('--visdom', action='store_true', default=False)
    args = parser.parse_args()
    return args


args = parse_args()
cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

if args.visdom == True:
    viz = visdom.Visdom()

categories = parse_categories(parse_data(args.data)['names'])

# Load model
cfg = darktorch.utils.parse_configuration(args.cfg)
model = darktorch.nn.Darknet(cfg)
darktorch.utils.init_layers(model, cfg)
model.load_weights(args.weights)
model.to(device)
print(model)

# Load the input image
w, h = int(cfg[0]['width']), int(cfg[0]['height'])
transform = transforms.Compose(
    [LetterboxDetect(w=w, h=h),
     BGR2RGB(),
     transforms.ToTensor()])
im = cv2.imread(args.image)

with torch.no_grad():

    model.eval()
    data = transform(im)
    data.unsqueeze_(0)
    data = data.to(device)

    t0 = time.time()
    
    output = model(data)
    detections = calculate_detections(model, output, im.shape[1], im.shape[0], w, h)
    detections = nonmax_suppression(detections, threshold=0.4)
    
    t1 = time.time()
    print('inference + detections time:', t1 - t0)
    print(detections)
    image = draw_detections_opencv(im, detections[0], categories)

    if args.visdom == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        viz.matplot(plt)
    else:
        cv2.imwrite('predictions.jpg', image)

