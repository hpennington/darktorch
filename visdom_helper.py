import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import visdom

im = cv2.imread(sys.argv[1])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

vis = visdom.Visdom()

plt.imshow(im)       
vis.matplot(plt)