import os
import sys
import pdb
import numpy as np
import cv2
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import visdom
from torchvision import transforms
from torch.utils.data import Dataset

import darktorch


class ListDataset(Dataset):
    def __init__(self,
                 filepath,
                 fp_transform_fn,
                 transform=None,
                 target_transform=None,
                 shuffle_labels=True):

        self.fp_transform_fn = fp_transform_fn
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle_labels = shuffle_labels

        with open(filepath, 'r') as f:
            self.image_paths = f.read().splitlines()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = cv2.imread(img_path)
        label_path = self.fp_transform_fn(img_path)
        label = np.loadtxt(label_path).reshape((-1, 5))

        if self.shuffle_labels == True:
            np.random.shuffle(label)

        if self.transform is not None:
            img = self.transform(img)

        #darktorch.utils.write_tensor(img, 'sized-{}.bin'.format(index), True, append=False)
        #plt.imshow(img.numpy().transpose([1, 2, 0]))
        #self.vis.matplot(plt)
        # pdb.set_trace()
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
