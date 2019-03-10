import io
import os
import pdb
import random
import torch
import numpy as np
import cv2
from PIL import Image
from torch.multiprocessing import Lock
from torchvision.transforms import functional as F


class BiTransform:
    def __init__(self):
        self.pid_contexts = {}

    def __call__(self, data):
        pid = os.getpid()

        # Label
        if data.shape[0] < 50:
        # if isinstance(data, np.ndarray):
            ctx = self.pid_contexts.get(pid)
            data = self.transform_label(ctx, data)

        # Image
        else:
        # elif isinstance(data, Image.Image):
            ctx = {}
            data = self.transform_image(ctx, data)
            self.pid_contexts[pid] = ctx

        return data

    def transform_image(self, ctx, img):
        raise NotImplementedError()

    def transform_label(self, ctx, label):
        raise NotImplementedError()


class RandomHorizontalFlip(BiTransform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def transform_image(self, ctx, img):
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            ctx['horizontal_flip'] = True

        return img

    def transform_label(self, ctx, label):
        if ctx.get('horizontal_flip') == True:
            label[:, 1] = 1.0 - label[:, 1]

        return label


class CustomToTensor:
    def __call__(self, data):
        return torch.from_numpy(data)


class Int2Float:
    def __call__(self, data):
        data = (data/255.).astype(np.float32)
        return data


class HWC2CHW:
    def __call__(self, data):
        return data.transpose([2, 0, 1])


class BGR2RGB:
    def __call__(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

def get_pixel_vectorized(im, y, x, c):
    h, w = im.shape[0], im.shape[1]
    pixel_vec = np.zeros(x.shape).astype(np.float32)
    mask = (x < w) & (y < h)
    pixel_vec[mask] = im[y[mask], x[mask], c]
    return pixel_vec

def bilinear_interpolate_vectorized(im, x, y, c):
    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    dx = x - ix
    dy = y - iy
    # px0 = get_pixel_vectorized(im, iy, ix, c)
    # fname = 'px0.txtlog'

    # try:
    #     # prev_px0 = np.fromfile(fname).astype(np.float32)
    #     prev_px0 = np.loadtxt(fname).astype(np.float32)
    #     inter = np.concatenate([prev_px0, px0])
    #     # print(px0.dtype)
    #     # pdb.set_trace()
    #     # inter.tofile(fname)
    #     np.savetxt(fname, inter, fmt='%f')
    # except IOError as e:
    #     print('Exception thrown! ', e)
    #     # px0.tofile(fname)
    #     np.savetxt(fname, px0, fmt='%f')
            

    vals = (1-dy)*(1-dx)*get_pixel_vectorized(im, iy, ix, c) \
         + dy*(1-dx)*get_pixel_vectorized(im, iy+1, ix, c) \
         + dx*(1-dy)*get_pixel_vectorized(im, iy, ix+1, c) \
         + dy*dx*get_pixel_vectorized(im, iy+1, ix+1, c)
    return vals

def resize_vectorized(src_im, size):
    src_h, src_w, _ = src_im.shape
    h, w = size
    dest = np.zeros((h, w, 3), dtype=src_im.dtype)
    for c in range(3):
        y = np.repeat(np.arange(h), w)
        x = np.resize(np.arange(w), w*h)
        ry = (y / h) * src_h
        rx = (x / w) * src_w
        vals = bilinear_interpolate_vectorized(src_im, rx, ry, c)
        dest[:,:,c] = vals.reshape(h, w)

    return dest

def place_image(orig, w, h, dx, dy, canvas):
    img = resize_vectorized(orig, (h, w))
    # img.tofile('vals.binlog')
    # exit(0)
    canvas[dy:dy+h, dx:dx+w] = img
    return canvas


class LetterboxTrain(BiTransform):
    def __init__(self, jitter, w=416, h=416, nonrandom=False):
        super().__init__()

        self.jitter = jitter
        self.w = w
        self.h = h
        self.scales = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        self.nonrandom = nonrandom

    def transform_image(self, ctx, img):
        h, w, c = img.shape

        dw = self.jitter * w
        dh = self.jitter * h
        new_ar = (w + random.uniform(-dw, dw)) \
               / (h + random.uniform(-dh, dh))
        scale = 1

        if new_ar < 1:
            nh = scale * self.h
            nw = nh * new_ar
        else:
            nw = scale * self.w
            nh = nw / new_ar

        if self.nonrandom == True:
            dx = 0.0
            dy = 0.0
        else:
            dx = random.uniform(0, self.w - nw)
            dy = random.uniform(0, self.h - nh)
        dy = int(dy)
        dx = int(dx)

        base_im = np.full((self.h, self.w, 3), .5, dtype=np.float32)
        img = place_image(img, int(nw), int(nh), dx, dy, base_im)

        ctx['dx'] = -dx / self.w
        ctx['dy'] = -dy / self.h
        ctx['sx'] = nw / self.w
        ctx['sy'] = nh / self.h
        return img

    def transform_label(self, ctx, label):
        dx, dy = ctx['dx'], ctx['dy']
        sx, sy = ctx['sx'], ctx['sy']

        for n in range(len(label)):
            box = label[n][1:]
            x, y, w, h = box

            left = x - w / 2
            right = x + w / 2
            top = y - h / 2
            bottom = y + h / 2

            box[0] = left * sx - dx
            box[1] = right * sx - dx
            box[2] = top * sy - dy
            box[3] = bottom * sy - dy
            np.clip(box, 0, 1)

            new_box = np.zeros(len(box))

            new_box[0] = (box[0] + box[1]) / 2
            new_box[1] = (box[2] + box[3]) / 2
            new_box[2] = (box[1] - box[0])
            new_box[3] = (box[3] - box[2])
            np.clip(new_box[2:], 0, 1)

            label[n][1:] = new_box

        return label


class LetterboxDetect:
    def __init__(self, w=416, h=416):
        self.w = w
        self.h = h

    def __call__(self, im):
        height = im.shape[0]
        width = im.shape[1]
        nw = width
        nh = height
        w = self.w
        h = self.h

        if w / width < h / height:
            nw = w
            nh = height * w / width
        else:
            nh = h
            nw = width * h / height

        im = cv2.resize(im, (int(nw), int(nh)))
        base_im = np.zeros((self.h, self.w, 3), np.uint8)
        base_im[:] = (128, 128, 128)
        base_im[int((h - nh) / 2):int((h - nh) / 2)+im.shape[0], int((w - nw) / 2):int((w - nw) / 2)+im.shape[1]] = im
        base_im

        return base_im


class CollateTransform:
    def __call__(self, label):
        n = 30
        x = torch.full((n, 5), -1.0)
        n = n if (label.shape[0] > n) else label.shape[0]
        x[:n] = torch.from_numpy(label[:n])
        return x

