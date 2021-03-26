#!/usr/bin/env python3

import numpy as np
import cv2
from torch.multiprocessing import Process, Queue, Lock

import sys
import time
import subprocess
import argparse

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F

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
        '--weights', type=str, default='weights/yolov2-voc.weights')
    parser.add_argument('--cfg', type=str, default='cfg/yolov2-voc.cfg')
    parser.add_argument('--data', type=str, default='cfg/voc.data')
    args = parser.parse_args()
    return args


def detect(frame_queue, preds_queue, lock, args):
    cfg = darktorch.utils.parse_configuration(args.cfg)

    # Load the input image
    w, h = int(cfg[0]['width']), int(cfg[0]['height'])
    transform = transforms.Compose(
        [LetterboxDetect(w=w, h=h), BGR2RGB(), transforms.ToTensor()])

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    model = darktorch.nn.Darknet(cfg)
    #model.share_memory()
    darktorch.utils.init_layers(model, cfg)
    model.load_weights(args.weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        while(True):
            lock.acquire()
            print('lock acquired')
            if not frame_queue.empty():
                print('here')
                frame = frame_queue.get()
                lock.release()
                print('lock released')
                data = transform(frame)
                data.unsqueeze_(0)
                #data = data.to(device)            
            
                t0 = time.time()
                print('pre model') 
                output = model(data)
                t1 = time.time()
                print('Forward pass:', t1 - t0)

                detections = calculate_detections(model, output, 
                    frame.shape[1], frame.shape[0], w, h)
                detections = nonmax_suppression(detections, threshold=0.3)

                t2 = time.time()
                print('Post precess:', t2 - t1)
                
                while not preds_queue.empty():
                    preds_queue.get()

                preds_queue.put(detections)
            else:
                print('there')
                lock.release()
            

def main():
    args = parse_args()
    categories = parse_categories(parse_data(args.data)['names'])

    cap = cv2.VideoCapture(0)
    frame_queue = Queue()
    preds_queue = Queue()
    cur_dets = None
    frame_lock = Lock()

    proc = Process(target=detect, args=(frame_queue, preds_queue, frame_lock, args))
    proc.start()

    try:        
        
        while(True):
            ret, frame = cap.read()
            frame_lock.acquire()
            #print('frame_lock acquired')
            while not frame_queue.empty():
                frame_queue.get()

            frame_queue.put(frame)
            frame_lock.release()
            #print('frame_lock released')

            if not preds_queue.empty():
                cur_dets = preds_queue.get()

            if cur_dets is not None and len(cur_dets) > 0:
                print(cur_dets)
                frame = draw_detections_opencv(frame, cur_dets[0], categories)
                #cur_dets = None

            cv2.imshow('frame', frame)
            cv2.waitKey(1)
                
    except KeyboardInterrupt:
        print('Interrupted')
        proc.join()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
