#!/usr/bin/env bash

max_batches=2000
cfg=cfg/yolov2-voc.cfg

for lr in 0.01 0.001 0.0001; do

sed -i'' "s/max_batches=.*/max_batches="$max_batches"/" $cfg
sed -i'' "s/learning_rate=.*/learning_rate="$lr"/" $cfg
python3 train.py

done
