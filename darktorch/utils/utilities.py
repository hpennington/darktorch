import os
import math
import time
import struct
import random

import cv2
import numpy as np
import imghdr
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Sampler
from torch import nn
from torchvision import transforms

import darktorch
import darktorch.nn as dnn


def write_tensor(x, path, binary=False, append=False):
    mode = 'a' if append == True else 'w'
    if binary == True:
        # Replace tofile method to have append option
        #x.numpy().tofile(path)
        f = open(path, mode + '+b')
        f.write(x.contiguous().view(-1).numpy().data)
        f.close()

    else:
        f = open(path, mode)
        x = x.contiguous().view(-1)
        for b in range(x.shape[0]):
            value = str(x[b].item())
            f.write(value + '\n')
        f.close()


def read_tensor(data, path):
    print('memory-mapping file:', path)
    shape = data.shape
    data = torch.FloatTensor(torch.FloatStorage.from_file(path, 
        size=data.flatten().shape[0])).view(shape)
    return data


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b

    conv_model.weight.data.copy_(
        torch.reshape(
            torch.from_numpy(buf[start:start + num_w]),
            (conv_model.weight.shape[0], conv_model.weight.shape[1],
             conv_model.weight.shape[2], conv_model.weight.shape[3])))

    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()

    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(
        torch.reshape(
            torch.from_numpy(buf[start:start + num_w]),
            (conv_model.weight.shape[0], conv_model.weight.shape[1],
             conv_model.weight.shape[2], conv_model.weight.shape[3])))
    start = start + num_w
    return start


def load_weights(net, fp):
    with open(fp, 'rb') as f:
        header = np.fromfile(f, count=3, dtype=np.int32)
        major, minor, revision = header

        if major * 10 + minor >= 2 and major < 1000 and minor < 1000:
            net.samples_processed = np.fromfile(
                f, count=1, dtype=np.int64).item()
        else:
            net.samples_processed = np.fromfile(
                f, count=1, dtype=np.int32).item()

        buf = np.fromfile(f, dtype=np.float32)

    start = 0
    layer_idx = 0
    module_list = list(net.children())

    while (start < len(buf) and layer_idx < len(module_list)):
        layer = module_list[layer_idx]
        if layer.__class__ is dnn.Conv:
            if hasattr(layer, 'batch_norm'):
                start = load_conv_bn(buf, start, layer.conv, layer.batch_norm)
            else:
                start = load_conv(buf, start, layer.conv)

        layer_idx += 1


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


def save_weights(net, fp, cutoff=0):
    if cutoff <= 0:
        cutoff = len(net.configuration) - 1

    f = open(fp, 'wb')
    major = 0
    minor = 1
    revision = 0
    n = net.samples_processed
    header = np.array([major, minor, revision, n], dtype=np.int32)
    header.tofile(f)
    module_list = list(net.modules())
    ind = 0
    for blockId in range(1, cutoff + 1):
        ind = ind + 1
        block = net.configuration[blockId]
        if block['type'] == 'convolutional':
            model = module_list[ind]
            batch_normalize = block.get('batch_normalize')
            batch_normalize = True if batch_normalize == '1' else False

            if batch_normalize == True:
                save_conv_bn(f, model.conv, model.bn)
            else:
                save_conv(f, model.conv)
        elif block['type'] == 'connected':
            model = module_list[ind]
            if block['activation'] != 'linear':
                save_fc(fc, model)
            else:
                save_fc(fc, model[0])
        else:
            pass

    f.close()


def init_layers(net, blocks):

    filters = 0
    filters_dim_cache = {}
    net.masks = None
    net.yolo_indices = None

    for index in range(-1, len(blocks) - 1):

        block = blocks[index + 1]
        block_type = block['type']

        if block_type == 'net':
            net.batch_size = int(block['batch'])
            net.subdivisions = int(block['subdivisions'])
            net.learning_rate = float(block['learning_rate'])
            net.channels = int(block['channels'])
            net.width = int(block['width'])
            net.height = int(block['height'])
            net.momentum = float(block['momentum'])
            net.weight_decay = float(block['decay'])
            net.max_batches = int(block['max_batches'])
            net.saturation = float(block['saturation'])
            net.hue = float(block['hue'])
            net.exposure = float(block['exposure'])
            net.steps = [int(x.strip()) for x in block['steps'].split(',')]
            net.scales = [float(x.strip()) for x in block['scales'].split(',')]
            net.burn_in = None

            filters = net.channels

            try:
                net.burn_in = int(block['burn_in'])
            except Exception as e:
                print(e)

        elif block_type == 'region':
            net.num_classes = int(block['classes'])
            net.num_priors = int(block['num'])
            net.jitter = float(block['jitter'])
            net.threshold = float(block['thresh'])
            net.multi_scale = True if int(block['random']) == 1 else False
            net.priors = np.reshape(
                [float(x.strip()) for x in block['anchors'].split(',')],
                [-1, 2])
            net.add_module(
                str(index), dnn.RegionLayer(net.num_classes, softmax=True))

        elif block_type == 'yolo':
            net.num_classes = int(block['classes'])
            net.num_priors = int(block['num'])
            net.jitter = float(block['jitter'])
            net.threshold = float(block['ignore_thresh'])
            net.truth_thresh = float(block['truth_thresh'])

            mask = [int(x.strip()) for x in block['mask'].split(',')]

            masks = net.masks if net.masks is not None else []
            masks.append(mask)
            net.masks = masks

            yolo_layers = net.yolo_indices if net.yolo_indices is not None else []
            yolo_layers.append(index)
            net.yolo_indices = yolo_layers

            net.multi_scale = True if int(block['random']) == 1 else False
            net.priors = np.reshape(
                [float(x.strip()) for x in block['anchors'].split(',')],
                [-1, 2])
            net.add_module(
                str(index), dnn.RegionLayer(net.num_classes, softmax=False))

        elif block_type == 'convolutional':
            s = int(block['stride'])
            f = int(block['filters'])
            k = int(block['size'])
            activation = block['activation']
            pad_true = int(block['pad'])

            pad = (k - 1) // 2 if pad_true else 0
            bn = True if ('batch_normalize' in block) \
            and int(block['batch_normalize']) == 1 else False

            net.add_module(
                str(index),
                dnn.Conv(
                    filters,
                    f,
                    k,
                    s,
                    pad,
                    batch_norm=bn,
                    activation=activation))

            filters = f
            filters_dim_cache[index] = filters

        elif block_type == 'shortcut':
            residual_layers = block['from'].split(',')
            activation = block['activation']

            net.add_module(
                str(index),
                dnn.Shortcut(net.output_cache, residual_layers, activation))

            filters_dim_cache[index] = filters

        elif block_type == 'maxpool':
            k = int(block['size'])
            s = int(block['stride'])

            if s == 1:
                net.add_module(
                    str(index),
                    dnn.MaxPoolStride1())
            else:
                net.add_module(
                    str(index), nn.MaxPool2d(kernel_size=k, stride=s))

        elif block_type == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + index for i in layers]

            if len(layers) == 1:
                filters = filters_dim_cache[layers[0]]
                filters_dim_cache[index] = filters

            elif len(layers) == 2:
                x1 = filters_dim_cache[layers[0]]
                x2 = filters_dim_cache[layers[1]]
                filters = x1 + x2
                filters_dim_cache[index] = filters

            net.add_module(str(index), dnn.Route(net.output_cache, layers))

        elif block_type == 'reorg':
            stride = int(block['stride'])
            net.add_module(str(index), dnn.Reorg(stride))
            filters = (stride * stride * filters)
            filters_dim_cache[index] = filters

        elif block_type == 'upsample':
            stride = int(block['stride'])
            net.add_module(str(index), nn.Upsample(scale_factor=stride))
            filters_dim_cache[index] = filters


def darknet_from_cfg(path):
    cfg = darktorch.utils.parse_configuration(path)
    net = darktorch.nn.Darknet(cfg)
    init_layers(net, cfg)
    return net


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = l2.clone()
    lmask = torch.tensor(l1 > l2, dtype=torch.uint8, device=x1.device) == True
    left[lmask] = l1[lmask]

    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = r2.clone()
    rmask = torch.tensor(r1 < r2, dtype=torch.uint8, device=x1.device) == True
    right[rmask] = r1[rmask]

    return right - left


def intersection(a, b):
    w = overlap(a[:, 0], a[:, 2], b[:, 0], b[:, 2])
    h = overlap(a[:, 1], a[:, 3], b[:, 1], b[:, 3])
    area = w * h
    mask = ((w < 0) + (h < 0) > 0)
    area[mask] = 0
    return area


def union(a, b):
    i = intersection(a, b)
    u = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - i
    return u


def calculate_iou(a, b):
    return intersection(a, b) / union(a, b)


def plot_detection(device, model, data, output, categories, writer, step):
    output_boxes = [
        get_region_boxes(batch, 0.5, model.num_classes, model.priors)[0]
        for batch in output
    ]

    output_boxes = [nonmax_suppression(batch, 0.4) for batch in output_boxes]

    ToPIL = transforms.ToPILImage()
    display_imgs = [ToPIL(img) for img in data.cpu()]

    for img_idx, img in enumerate(display_imgs):
        display_imgs[img_idx] = plot_boxes(
            img, output_boxes[img_idx], class_names=categories)

    ToTensor = transforms.ToTensor()
    display_imgs = [ToTensor(img) for img in display_imgs]
    display_imgs = vutils.make_grid(display_imgs, normalize=True, padding=10)
    writer.add_image('Train detection', display_imgs, step)


def plot_histogram(model, writer, step, plot_grads=False):
    for name, param in model.named_parameters():
        if 'bn' not in name:
            writer.add_histogram(name, param, step)
            if plot_grads == True and param.grad is not None:
                name = name + '_gradient'
                writer.add_histogram(name, param.grad, step)


def one_hot_encode(x, num_classes):
    vec = torch.zeros(num_classes)
    vec[int(x)] = 1
    return vec


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def _nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline=color)


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                                [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw_rectangle(draw, ((x1, y1), (x2, y2)), rgb, width=3)
        # draw.rectangle([x1, y1, x2, y2], outline = rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img

def plot_boxes_opencv(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                                [1, 1, 0], [1, 0, 0]])

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    
    for i in range(len(boxes)):
        box = boxes[i]
        # print('box:', box)
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        # if len(box) >= 7 and class_names:
        #     cls_conf = box[5]
        #     cls_id = box[6]
        #     print('%s: %f' % (class_names[cls_id], cls_conf))
        #     classes = len(class_names)
        #     offset = cls_id * 123457 % classes
        #     red = get_color(2, offset, classes)
        #     green = get_color(1, offset, classes)
        #     blue = get_color(0, offset, classes)
        #     rgb = (red, green, blue)
        #     draw.text((x1, y1), class_names[cls_id], fill=rgb)
        # draw_rectangle(draw, ((x1, y1), (x2, y2)), rgb, width=3)
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=rgb, thickness=2)
    return img

def _calculate_region_boxes(orig_w,
                            orig_h,
                            net_w,
                            net_h,
                            output,
                            anchors,
                            num_classes,
                            threshold,
                            v3=True):
    B, Z, H, W = output.shape
    N = len(anchors) // 2
    downscale_factor = net_h // H
    detections = []
    
    for b in range(B):
        for h in range(H):
            for w in range(W):
                for n in range(N):
                    
                    objectness = output[b, n * (Z // N) + 4, h, w]
                    if objectness > threshold:
                        bx = (w + output[b, n * (Z // N), h, w]) / W
                        by = (h + output[b, n * (Z // N) + 1, h, w]) / H

                        if v3 == True:
                            w_div = W * downscale_factor
                            h_div = H * downscale_factor
                        else:
                            w_div = W
                            h_div = H

                        bw = torch.exp(output[b, n * (Z // N) + 2, h, w]
                                       ) * anchors[2 * n] / w_div
                        bh = torch.exp(output[b, n * (Z // N) + 3, h, w]
                                       ) * anchors[2 * n + 1] / h_div

                        # "Correct yolo boxes"
                        if ((W * downscale_factor) / orig_w) < ((H * downscale_factor) / orig_h):
                            new_w = W * downscale_factor
                            new_h = (orig_h * (W * downscale_factor)) / orig_w
                        else:
                            new_h = H * downscale_factor
                            new_w = (orig_w * (H * downscale_factor)) / orig_h

                        bx = (bx - ((W * downscale_factor) - new_w) / 2 /
                              (W * downscale_factor)) / (new_w / (W * downscale_factor))
                        by = (by - ((H * downscale_factor) - new_h) / 2 /
                              (H * downscale_factor)) / (new_h / (H * downscale_factor))
                        bw *= (W * downscale_factor) / new_w
                        bh *= (H * downscale_factor) / new_h

                        max_prob = -1.
                        max_probi = -1
                        for c in range(num_classes):
                            prob = objectness * output[b, n * (Z // N) +
                                                       (5 + c), h, w]
                            if prob > max_prob:
                                max_prob = prob
                                max_probi = c

                        detection = [
                            bx.item(),
                            by.item(),
                            bw.item(),
                            bh.item(),
                            objectness.item(),
                            max_prob.item(), max_probi
                        ]
                        detections.append(detection)

    return detections


def calculate_region_boxes_v2(orig_w, orig_h, net_w, net_h, output, anchors, num_classes,
                              threshold):
    return _calculate_region_boxes(
        orig_w, orig_h, net_w, net_h, output, anchors, num_classes, threshold, v3=False)


def calculate_region_boxes_v3(orig_w, orig_h, net_w, net_h, output, anchors, num_classes,
                              threshold):
    return _calculate_region_boxes(
        orig_w, orig_h, net_w, net_h, output, anchors, num_classes, threshold, v3=True)


def calculate_detections(net, net_output, image_w, image_h, net_w, net_h):
    threshold = 0.5
    detections = []
    w, h = image_w, image_h

    # YOLO V3
    if net.masks is not None:
        for i, mask in enumerate(net.masks):
            yolo_layer_i = net.yolo_indices[i]
            anchors = np.reshape([net.priors[e] for e in mask], -1)

            output = net.output_cache[yolo_layer_i]
            region_boxes = calculate_region_boxes_v3(
                w, h, net_w, net_h, output, anchors, net.num_classes, threshold)

            if len(region_boxes) > 0:
                if len(detections) == 0:
                    detections.append(region_boxes)
                elif len(detections) == 1:
                    detections[0].extend(region_boxes)

    # YOLO V2
    else:
        anchors = np.reshape(net.priors, -1)
        region_boxes = calculate_region_boxes_v2(
            w, h, net_w, net_h, net_output, anchors, net.num_classes, net.threshold)
        if len(region_boxes) > 0:
            detections.append(region_boxes)

    return detections

def nonmax_suppression(detections, threshold):
    detections = [_nms(batch, threshold) for batch in detections]
    return detections

def draw_detections_opencv(image, detections, categories):
    image = plot_boxes_opencv(image, detections, class_names=categories)
    return image

def draw_detections(image_tensor, detections, categories):
    ToPIL = transforms.ToPILImage()
    display_images = [ToPIL(image) for image in image_tensor]

    for i, image in enumerate(display_images):
        display_images[i] = plot_boxes(
            image, detections[i], class_names=categories)

    return display_images


class LRScheduler():
    def __init__(self, optimizer, init_lr, steps, scales, burn_in=None):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.steps = steps
        self.scales = scales
        self.burn_in = burn_in

        self.lr = init_lr

    def update(self, iteration):
        alpha = self.calculate_alpha(iteration)
        if alpha != self.lr:
            self.set_alpha(alpha)

    def calculate_alpha(self, iteration):
        alpha = self.init_lr

        for index, step in enumerate(self.steps):
            if iteration > step:
                alpha *= self.scales[index]

        if self.burn_in is not None:
            if iteration <= self.burn_in:
                alpha *= 0.1

        return alpha

    def set_alpha(self, lr):
        print('Learning rate set to:', lr)
        self.lr = lr
        for group in self.optimizer.param_groups:
            group['lr'] = lr
