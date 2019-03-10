import numpy as np
import torch
from torch.utils.data import Sampler


class MultiScaleSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.adjust_length = False

    def complete_epoch_samples(self, n):
        self.adjust_length = True
        self.n = int(n)

    def __iter__(self):
        n = len(self.data_source)
        if self.adjust_length:
            n = self.n

        if self.shuffle == True:
            item = torch.randperm(n).tolist()
        else:
            item = range(n)

        return iter(item)

    def __len__(self):
        if self.adjust_length == True:
            self.adjust_length = False
            return self.n

        return len(self.data_source)


def list_collate(batch):
    batch = list(map(list, zip(*batch)))  # transpose list of list
    batch[0] = torch.stack(batch[0])
    return batch

def shit_collate(batch):
    return batch