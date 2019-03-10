import os
import pdb
import math
import time

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darktorch.utils import calculate_iou, one_hot_encode, write_tensor


class RegionLossV3(nn.Module):
    def __init__(self, priors, threshold=0.5, n_classes=20):
        super().__init__()
        self.priors = priors
        self.threshold = threshold
        self.n_classes = n_classes
        self.obj_scale = 5
        self.nobj_scale = 1

        self.biases = None
        self.seen = 0

    def forward(self, output, target):
        tmp_target = target.cpu()
        tmp_target[tmp_target == -1] = 0
        write_tensor(tmp_target, 'target.binlog', False, False)
        
        K = 5 + self.n_classes
        B, Z, H, W = output.shape
        N = Z // K
        T = target.shape[1]
        scale_factor = 32

        output = output.permute((0, 2, 3, 1)).view(B, H, W, N, K).contiguous()
        deltas = torch.zeros(output.shape, device=output.device)

        deltas[:, :, :, :, 4] = self.nobj_scale * (0 - output[:, :, :, :, 4])

        # avg_anyobj = output[:, :, :, :, 4].sum() / (H * W * N * B)
        
        # if self.biases is None:
        #     self.biases = self.calculate_biases(B, H, W, N).to(output.device)

        # output_postp = output[:, :, :, :, :4].clone().detach()
        # print('xval:', output[0, 0, 0, 0, :4])
        # output_postp[:, :, :, :, 0] = (
        #     self.biases[:, :, :, :, 0] + output_postp[:, :, :, :, 0]) / (W*scale_factor)
        # output_postp[:, :, :, :, 1] = (
        #     self.biases[:, :, :, :, 1] + output_postp[:, :, :, :, 1]) / (H*scale_factor)
        # output_postp[:, :, :, :, 2] = (self.biases[:, :, :, :, 2] * torch.exp(
        #     output_postp[:, :, :, :, 2])) / W
        # output_postp[:, :, :, :, 3] = (self.biases[:, :, :, :, 3] * torch.exp(
        #     output_postp[:, :, :, :, 3])) / H
        # # write_tensor(output_postp.cpu(), 'preds.binlog', True)
        # # for e in output_postp.view(-1, 4):
        #     # print('x:{:.6f}, y:{:.6f}, w:{:.6f}, h:{:.6f}'.format(e[0], e[1], e[2], e[3]))
        
        # # # print(output_postp)
        # iou_target = target[:, :, 1:].clone().detach()
        # iou_target = iou_target.unsqueeze(1).expand(-1, H * W * N, -1,
        #                                             -1).contiguous()

        # ious = calculate_iou(
        #     output_postp.unsqueeze(4).expand(-1, -1, -1, -1, T,
        #                                      -1).contiguous().view(-1, 4),
        #     iou_target.view(-1, 4)).view(B, H, W, N, T)

        # iou_mask = ious.max(dim=4)[0] > 0.7
        # print(ious[iou_mask])
        # d_mask = torch.zeros(iou_mask.unsqueeze(4).shape).byte().to(
        #     output.device).expand(-1, -1, -1, -1, K)
        # d_mask[:, :, :, :, 4] = iou_mask

        # deltas[d_mask] = 0
        # print(d_mask)
        write_tensor(deltas.detach().cpu(), 'deltas.binlog', True)
        exit(0)
        return output.mean()

    def apply_sort_args(self, x, args, length):
        x2 = torch.zeros(x.shape)
        for index, arg in enumerate(args):
            x2[length * index:length * index +
               length] = x[arg * length:length * arg + length]
        return x2

    def calculate_biases(self, batch, height, width, n_anchors):
        biases = torch.zeros((batch, height, width, n_anchors, 4))
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    for n in range(n_anchors):
                        for i in range(4):
                            if i < 2:
                                biases[b, h, w, n, i] = w if i == 0 else h
                            else:
                                biases[b, h, w, n, i] = self.priors[2 * n +
                                                                    (i - 2)]
        return biases


class RegionLossV2(nn.Module):
    def __init__(self, priors, threshold=0.5, n_classes=20):
        super().__init__()

        self.priors = priors
        self.threshold = threshold
        self.n_classes = n_classes
        self.obj_scale = 5
        self.nobj_scale = 1

        self.biases = None
        self.seen = 0

    def forward(self, output, target):
        # print('output shape', output.shape)
        # print('target shape', target.shape)
        # print(target)

        K = 5 + self.n_classes
        B, Z, H, W = output.shape
        N = Z // K
        T = target.shape[1]

        output = output.permute((0, 2, 3, 1)).view(B, H, W, N, K).contiguous()
        deltas = torch.zeros(output.shape, device=output.device)
        # write_tensor(deltas.detach().cpu(), 'deltas.bin', True)

        deltas[:, :, :, :, 4] = self.nobj_scale * (0 - output[:, :, :, :, 4])
        # write_tensor(deltas.detach().cpu(), 'deltas.bin', True)
        avg_anyobj = output[:, :, :, :, 4].sum() / (H * W * N * B)

        if self.biases is None:
            self.biases = self.calculate_biases(B, H, W, N).to(output.device)

        output_postp = output[:, :, :, :, :4].clone().detach()
        output_postp[:, :, :, :, 0] = (
            self.biases[:, :, :, :, 0] + output_postp[:, :, :, :, 0]) / W
        output_postp[:, :, :, :, 1] = (
            self.biases[:, :, :, :, 1] + output_postp[:, :, :, :, 1]) / H
        output_postp[:, :, :, :, 2] = (self.biases[:, :, :, :, 2] * torch.exp(
            output_postp[:, :, :, :, 2])) / W
        output_postp[:, :, :, :, 3] = (self.biases[:, :, :, :, 3] * torch.exp(
            output_postp[:, :, :, :, 3])) / H

        iou_target = target[:, :, 1:].clone().detach()
        iou_target = iou_target.unsqueeze(1).expand(-1, H * W * N, -1,
                                                    -1).contiguous()

        ious = calculate_iou(
            output_postp.unsqueeze(4).expand(-1, -1, -1, -1, T,
                                             -1).contiguous().view(-1, 4),
            iou_target.view(-1, 4)).view(B, H, W, N, T)

        iou_mask = ious.max(dim=4)[0] > 0.6

        d_mask = torch.zeros(iou_mask.unsqueeze(4).shape).byte().to(
            output.device).expand(-1, -1, -1, -1, K)
        d_mask[:, :, :, :, 4] = iou_mask

        deltas[d_mask] = 0

        if self.seen < 12800:
            pass

        output = output.cpu()
        deltas = deltas.cpu()
        target = target.cpu()
        output_postp = output_postp.cpu()

        # write_tensor(deltas.detach(), 'deltas.bin', True)

        truth_boxes = target[:, :, 1:].clone().detach()
        truth_classes = target[:, :, 0].clone().detach()
        i = (truth_boxes[:, :, 0] * W).long()
        j = (truth_boxes[:, :, 1] * H).long()

        tbs_shifted = truth_boxes.clone()
        tbs_shifted[:, :, 0] = 0
        tbs_shifted[:, :, 1] = 0

        shift_mask = torch.zeros((B, T, H, W)).byte()

        for b in range(B):
            for t in range(T):
                shift_mask[b, t, j[b, t], i[b, t]] = 1

        output_shifted = output_postp.unsqueeze(1).expand(
            -1, T, -1, -1, -1, -1)[shift_mask].view(B, T, N, 4)
        output_shifted[:, :, :, :2] = 0

        bias_match = False
        if bias_match == True:
            pass

        ious = calculate_iou(
            tbs_shifted.unsqueeze(2).expand(-1, -1, 5, -1).contiguous().view(
                -1, 4), output_shifted.view(-1, 4)).view(B, T, 5)

        max_ious, max_indices = ious.max(dim=2)

        priors = torch.from_numpy(np.array(self.priors)).view(N, 2)
        priors_volume = torch.zeros((B, T, 2))

        for b in range(B):
            for t in range(T):
                if max_ious[b, t] != 0.0:
                    priors_volume[b, t] = priors[max_indices[b, t]]

        tx = truth_boxes[:, :, 0] * W
        ty = truth_boxes[:, :, 1] * H
        tx = tx - i.float()
        ty = ty - j.float()

        tw = torch.log(truth_boxes[:, :, 2] * W / priors_volume[:, :, 0])
        th = torch.log(truth_boxes[:, :, 3] * H / priors_volume[:, :, 1])
        tw[torch.isnan(tw)] = 0.0
        th[torch.isnan(th)] = 0.0

        scales = (1.0 * (2 - truth_boxes[:, :, 2] * truth_boxes[:, :, 3])
                  ).unsqueeze(2).expand(-1, -1, 4)
        t_mask = scales != 1.0
        p_mask = torch.zeros(deltas.shape).byte()
        o_mask = torch.zeros(deltas.shape).byte()
        c_mask = torch.zeros(deltas.shape).byte()

        for b in range(B):

            duplicate_dict = {}
            for t in range(T):
                if tx[b, t] == 0.0 and ty[b, t] == 0.0 and tw[
                        b, t] == 0.0 and th[b, t] == 0.0:
                    break
                cur_j = j[b, t].item()
                cur_i = i[b, t].item()
                max_n = max_indices[b, t].item()
                p_mask[b, cur_j, cur_i, max_n, :4] = 1
                o_mask[b, cur_j, cur_i, max_n, 4] = 1
                c_mask[b, cur_j, cur_i, max_n, 5:] = 1

                if tuple((cur_i, cur_j, max_n)) in duplicate_dict:
                    duplicate_dict[tuple((cur_i, cur_j, max_n))].append((b, t))
                else:
                    duplicate_dict[tuple((cur_i, cur_j, max_n))] = [(b, t)]

            for key in duplicate_dict.keys():
                items = duplicate_dict[key]
                if len(items) > 1:
                    for item in items[:-1]:
                        truth_boxes[item[0]][item[1]] = -1
                        target[item[0], item[1]] = -1
                        truth_classes[item[0]][item[1]] = -1
                        t_mask[item[0]][item[1]] = 0

        t_volume = torch.stack([tx, ty, tw, th], dim=2)
        t_volume = t_volume[t_mask]
        scales = scales[t_mask]

        # Sort t_volume & scales
        sorted_t_volume = torch.zeros(t_volume.shape)
        sorted_scales = torch.zeros(scales.shape)
        sorted_truth_classes = torch.zeros(
            truth_classes[truth_classes > -1].shape)
        sorted_t_i = 0

        for b in range(B):
            max_indices_batch = max_indices[b][target[b, :, 1] > -1]
            ts = truth_boxes[b][truth_boxes[b] > -1].view(-1, 4)
            i_s = (ts[:, 0] * W).long()
            j_s = (ts[:, 1] * H).long()
            ps = list(zip(j_s, i_s, max_indices_batch))
            sort_args = list(
                OrderedDict(sorted(enumerate(ps), key=lambda x: x[1])).keys())
            sub_sorted_t = self.apply_sort_args(
                t_volume[sorted_t_i * 4:4 * (sorted_t_i + len(sort_args))],
                sort_args, 4)
            sub_sorted_s = self.apply_sort_args(
                scales[sorted_t_i * 4:4 * (sorted_t_i + len(sort_args))],
                sort_args, 4)
            sub_sorted_c = self.apply_sort_args(
                truth_classes[b][truth_classes[b] > -1], sort_args, 1)
            sorted_t_volume[sorted_t_i * 4:4 *
                            (sorted_t_i + len(sort_args))] = sub_sorted_t
            sorted_scales[sorted_t_i * 4:4 *
                          (sorted_t_i + len(sort_args))] = sub_sorted_s
            sorted_truth_classes[sorted_t_i:(
                sorted_t_i + len(sort_args))] = sub_sorted_c
            sorted_t_i += len(sort_args)

        deltas[p_mask] = sorted_scales * (sorted_t_volume - output[p_mask])
        deltas[o_mask] = 5 * (1 - output[o_mask])

        rescore = False
        if rescore == True:
            pass

        background = False
        if background == True:
            deltas[o_mask] = 5 * (0 - output[o_mask])

        class_target = torch.zeros(output[c_mask].shape)
        class_scale = torch.tensor(1.0)

        ci = 0
        for cat in sorted_truth_classes:
            v = one_hot_encode(cat, self.n_classes)
            class_target[ci:ci + self.n_classes] = v
            ci += self.n_classes

        deltas[c_mask] = class_scale * (class_target - output[c_mask])
        cost = torch.sqrt((deltas**2).sum())**2

        # print('REGION LOSS:', cost)

        # print('Region Avg IOU: {}, Class: {},\n Obj: {}, No Obj: {},\n Avg Recall: {}, count: {}'.format(
        #     avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj, recall/count, count))

        return cost

    def apply_sort_args(self, x, args, length):
        x2 = torch.zeros(x.shape)
        for index, arg in enumerate(args):
            x2[length * index:length * index +
               length] = x[arg * length:length * arg + length]
        return x2

    def calculate_biases(self, batch, height, width, n_anchors):
        biases = torch.zeros((batch, height, width, n_anchors, 4))
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    for n in range(n_anchors):
                        for i in range(4):
                            if i < 2:
                                biases[b, h, w, n, i] = w if i == 0 else h
                            else:
                                biases[b, h, w, n, i] = self.priors[2 * n +
                                                                    (i - 2)]
        return biases
