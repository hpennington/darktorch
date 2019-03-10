
# Quick and dirty end-to-end integration tests
# Run 'python -m pytest' in the root directory
#

import os
import sys
import re
import time
import collections
import argparse
import random
import subprocess

import numpy as np


class STDOUTParserError(Exception):
    def __init__(self, message):
        self.message = message


class STDOUTDetectionParser():
    def parse(self, data):
        data = data.decode()
        match = re.search(r"\[\[.*\]\]", data).group(0).replace('[', '').replace(']', '').replace(',', '')

        if match is None:
            raise STDOUTParserError('No regex matches found')
        
        try:
            detections = [float(s) for s in match.split()]
            detections = np.array(detections).reshape(-1, 7)
        except:
            raise STDOUTParserError('reshape failed')
        
        return detections


class STDOUTLossParser():
    def parse(self, data):
        data = data.decode()
        match = re.search(r"iter.*epoch.*loss.*", data).group(0)
        
        if match is None:
            raise STDOUTParserError('No regex matches found')

        match = re.findall("\d+\.\d+", match)
        
        if match is None:
            raise STDOUTParserError('No regex matches found')
        
        try:
            loss = float(match[0])
        except:
            raise STDOUTParserError('Parse to float failed')
        
        return loss 


def compare_detection(data_file, cfg_file, weights_file, im_file, ref_file,
                tol=2e-2):
    args = ['python', 'detect.py', '--cfg', cfg_file, 
            '--data', data_file, '--image', 
            im_file, '--weights', weights_file]

    try:
        cproc = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as err:
        assert False, err.stderr

    try:
        detections = STDOUTDetectionParser().parse(cproc.stdout)
    except STDOUTParserError as err:
        assert False, err

    try:
        # Load pre-computed Darknet framework detections
        detections_ref = np.fromfile(
            ref_file, sep=' ', dtype=np.float32).reshape(-1, 7)
    except:
        assert False, 'STDOUTDetectionParser failed!'

    assert np.allclose(detections_ref, detections, rtol=tol, atol=tol)

def compare_loss(data_file, cfg_file, weights_file, ref_file,
                tol=2e-2):
    args = ['python', 'train.py',
            '--cfg', cfg_file, 
            '--data', data_file, 
            '--weights', weights_file,
            '--num-workers=0',
            '--nonrandom',
            '--no-shuffle',
            '--once']

    try:
        cproc = subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as err:
        assert False, err.stderr

    try:
        loss = STDOUTLossParser().parse(cproc.stdout)
    except STDOUTParserError as err:
        assert False, err

    try:
        # Load pre-computed Darknet framework loss
        loss_ref = np.fromfile(
            ref_file, sep=' ', dtype=np.float32)[0]
    except:
        assert False, 'STDOUTLossParser failed!'

    assert np.isclose(loss, loss_ref, rtol=tol, atol=tol)

def test_yolov2_voc():
    compare_detection('test/test-files/voc.data', 
                'test/test-files/yolov2-voc.cfg',
                'weights/yolov2-voc.weights', 'data/dog.jpg',
                'test/test-files/yolov2-voc-darknet-results.txt')

def test_yolov2_coco():
    compare_detection('test/test-files/coco.data', 
                'test/test-files/yolov2.cfg',
                'weights/yolov2.weights', 'data/person.jpg',
                'test/test-files/yolov2-coco-darknet-results.txt')

def test_yolov2_tiny():
    compare_detection('test/test-files/coco.data', 
                'test/test-files/yolov2-tiny.cfg',
                'weights/yolov2-tiny.weights', 'data/person.jpg',
                'test/test-files/yolov2-tiny-darknet-results.txt')

def test_yolov3_coco():
    compare_detection(
        'test/test-files/coco.data',
        'test/test-files/yolov3.cfg',
        'weights/yolov3.weights',
        'data/dog.jpg',
        'test/test-files/yolov3-darknet-results.txt',
        tol=4e-2)

def test_yolov3_tiny():
    compare_detection('test/test-files/coco.data', 
                'test/test-files/yolov3-tiny.cfg',
                'weights/yolov3-tiny.weights', 'data/giraffe.jpg',
                'test/test-files/yolov3-tiny-darknet-results.txt')

def test_yolov2_voc_loss():
    compare_loss('test/test-files/voc.data', 
                'test/test-files/yolov2-voc.cfg',
                'weights/yolov2-voc.weights', 
                'test/test-files/yolov2-voc-darknet-loss-results.txt')

def test_yolov2_tiny_voc_loss():
    compare_loss('test/test-files/voc.data', 
                'test/test-files/yolov2-tiny-voc.cfg',
                'weights/yolov2-tiny-voc.weights', 
                'test/test-files/yolov2-tiny-voc-darknet-loss-results.txt')

def test_yolov2_coco_loss():
    compare_loss('test/test-files/coco.data', 
                'test/test-files/yolov2.cfg',
                'weights/yolov2.weights', 
                'test/test-files/yolov2-coco-darknet-loss-results.txt')

def test_yolov2_tiny_coco_loss():
    compare_loss('test/test-files/coco.data', 
                'test/test-files/yolov2-tiny.cfg',
                'weights/yolov2-tiny.weights', 
                'test/test-files/yolov2-tiny-coco-darknet-loss-results.txt')