# DarkTorch

###### Designed as a drop in replacement for the YOLO Darknet framework written in idiomatic PyTorch.

###### This project was written both as a learning experience and to make hacking on YOLO easier than the alternative C framework.

###### Good luck!

## Setup
Use conda for easiest setup

```
conda env create -f environment.yml
```
Required packages if not using conda:

- pytorch
- torchvision
- opencv-python

Optional packages:

- visdom
- matplotlib
- pytest

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