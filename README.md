![alt text](https://github.com/hpennington/darktorch/raw/master/abbey.jpg "The Beatles Abbey Road")

# DarkTorch

###### Designed as a drop in replacement for the YOLO Darknet framework written in idiomatic PyTorch.

###### This project was written both as a learning experience and to make hacking on YOLO easier than the alternative C framework.

###### Good luck!

##### Supports  YOLOv2 with YOLOv3 support coming soon!

## Setup
Use conda for easiest setup

```
conda env create -f environment.yml
conda activate darktorch
```
Required packages if not using conda:

- pytorch
- torchvision
- opencv-python

Optional packages:

- visdom
- matplotlib
- pytest

## Training

### Training on VOC
```
# Download VOC dataset
cd data
cp ../scripts/get_voc_dataset.sh ./
bash get_voc_dataset.sh

# Label VOC dataset
cp ../../scripts/voc_label.py
python3 voc_label.py
cd ../..

python3 train.py

```
### Training on COCO
```
# Download COCO dataset
cd data
cp ../scripts/get_coco_dataset.sh ./
bash get_coco_dataset.sh

cd ..

python3 train.py --cfg=cfg/yolov2-coco.cfg --weights=weights/darknet19_448.conv.23

```

### Training arguments

- --no-cuda
- --num-workers
- --clipping-norm
- --cfg
- --data
- --weights
- --no-shuffle
- --non-random
- --fintune
- --once

## Running the test suite
### Download both the COCO and VOC datasets

```
# From the root directory run:
cd data
cp ../scripts/get_voc_dataset.sh ./
cp ../scripts/get_coco_dataset.sh ./
bash get_voc_dataset.sh
cd ..
bash get_coco_dataset.sh
cd ..

# Run the label VOC script
cd voc
cp ../../scripts/voc_label.py
python3 voc_label.py
cd ../..

# Run the test suite
python3 -m pytest

```

## Citations
```
@misc{darknet13,
  author =   {Joseph Redmon},
  title =    {Darknet: Open Source Neural Networks in C},
  howpublished = {\url{http://pjreddie.com/darknet/}},
  year = {2013--2016}
}
```