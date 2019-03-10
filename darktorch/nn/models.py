import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import darktorch


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x


class Darknet(nn.Module):
    def __init__(self, configuration):
        super(Darknet, self).__init__()
        self.configuration = configuration
        self.output_cache = {}
        self.samples_processed = 0

    def forward(self, x):
        for i, module in enumerate(self.children()):
            x = module(x)
            # darktorch.utils.write_tensor(x.cpu().detach(), 'layer{:03d}.bin'.format(i), True)
            self.output_cache[i] = x
        self.samples_processed += x.shape[0]
        return x

    def load_weights(self, path):
        if '.pt' in path:
            self.load_state_dict(torch.load(path))
        else:
            darktorch.utils.load_weights(self, path)

    def save_weights(self, path):
        if '.pt' in path:
            torch.save(self.state_dict(), path)
        else:
            darktorch.utils.save_weights(self, path)


class Shortcut(nn.Module):
    def __init__(self, output_cache, residual_layers, activation):
        super().__init__()

        self.output_cache = output_cache
        self.residual_layers = residual_layers

        if activation == 'linear':
            activation = self.linear_activation

        self.activation = activation

    def forward(self, x):
        output_cache = [t[1] for t in self.output_cache.items()]
        for res in self.residual_layers:
            x += output_cache[int(res)]

        x = self.activation(x)
        return x

    def linear_activation(self, x):
        return x


class Route(nn.Module):
    def __init__(self, output_cache, layers):
        super(Route, self).__init__()
        self.output_cache = output_cache
        self.layers = layers

    def forward(self, x):
        if len(self.layers) == 1:
            x = self.output_cache[self.layers[0]]
        elif len(self.layers) == 2:
            x1 = self.output_cache[self.layers[0]]
            x2 = self.output_cache[self.layers[1]]
            x = torch.cat((x1, x2), 1)
        return x

    def extra_repr(self):
        s = str()
        for i in range(len(self.layers)):
            s += str(self.layers[i])
            if i != len(self.layers) - 1:
                s += ' '
        return s


class Reorg(nn.Module):
    def __init__(self, stride):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.data.shape
        stride = self.stride
        out_w = w // stride
        out_h = h // stride

        x = x.view(b, c // (stride**2), h, stride, w, stride).contiguous()
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(b, -1, out_h, out_w)
        return x


class RegionLayer(nn.Module):
    def __init__(self, n_classes, softmax=False):
        super().__init__()
        self.n_classes = n_classes
        self.softmax = softmax

    def forward(self, x):
        y = x.clone()
        y = y.permute((0, 2, 3, 1))
        B, H, W, z = y.shape
        nC = self.n_classes
        nA = z // (nC + 5)
        y = y.view((B, H, W, nA, (nC + 5)))

        y[:, :, :, :, :2] = x.permute((0, 2, 3, 1)).view(
            (B, H, W, nA, (nC + 5)))[:, :, :, :, :2].sigmoid()
        y[:, :, :, :, 4] = x.permute((0, 2, 3, 1)).view(
            (B, H, W, nA, (nC + 5)))[:, :, :, :, 4].sigmoid()

        if self.softmax == True:
            y[:, :, :, :, 5:] = nn.functional.softmax(
                x.permute((0, 2, 3, 1)).view((B, H, W, nA,
                                              (nC + 5)))[:, :, :, :, 5:],
                dim=4)
        else:
            y[:, :, :, :, 5:] = x.permute((0, 2, 3, 1)).view(
                (B, H, W, nA, (nC + 5)))[:, :, :, :, 5:].sigmoid()

        y = y.view((B, H, W, z)).permute((0, 3, 1, 2))
        return y


class Linear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Conv(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 k,
                 stride,
                 pad,
                 batch_norm=True,
                 activation='leaky'):
        super().__init__()

        def extra_conv():
            return conv_repr
        
        def extra_bn():
            return bn_repr.replace(', affine=True, track_running_stats=True', '')

        self.activation_type = activation

        self.use_batch_norm = batch_norm

        self.conv = nn.Conv2d(
            in_filters,
            out_filters,
            kernel_size=k,
            padding=pad,
            stride=stride,
            bias=not self.use_batch_norm)

        conv_repr = self.conv.__repr__()
        conv_repr = re.search(r'\((.*?)\)',conv_repr).group(1)
        self.conv.extra_repr = extra_conv

        if self.use_batch_norm == True:
            self.batch_norm = torch.nn.BatchNorm2d(out_filters)
            bn_repr = self.batch_norm.__repr__()
            bn_repr = re.search(r'\((.*?)\)',bn_repr).group(1)
            self.batch_norm.extra_repr = extra_bn

        if self.activation_type == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.activation = Linear()

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm == True:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x
