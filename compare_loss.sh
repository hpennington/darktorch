#!/usr/bin/env bash

DATA=test/test-files/coco.data
CFG=cfg/yolov3.cfg
WEIGHTS=weights/yolov3.weights

# tmux_execute <window> <cmd>
# Non-blocking execute command in tmux window
function tmux_execute {
    session=$1
    message=$2
    tmux send-keys -t $session "$message" C-m
}

# tmux_execute_wait <window> <cmd>
# Execute command in tmux window
function tmux_execute_wait {
    session=$1
    message=$2
    uuid=$(uuidgen)
    tmux send-keys -t $session \
        "$message; tmux wait-for -S $uuid" C-m; tmux wait-for $uuid 
}

# tmux_create_window <window> <init script>
# Create window, if window does not exist.
function tmux_create_window {
    title=$1
    init_cmd=$2
    if ! tmux list-windows | grep -q $title; then
        tmux new-window -d -n $title 
        tmux_execute $title "$init_cmd"
    fi
}

# Remove files from previous run
rm -f *.binlog *.txtlog
rm -f darknet/*.binlog darknet/*.txtlog

tmux_create_window darktorch "source activate darktorch"
tmux_create_window darknet "cd darknet"

tmux_execute_wait darktorch \
    "python train.py \
        --num-workers=0 \
        --cfg=$CFG \
        --data=$DATA \
        --weights=$WEIGHTS \
        --nonrandom \
        --once \
        --no-shuffle"

tmux_execute_wait darknet \
    "make -j8 && ./darknet detector train $DATA ../$CFG ../$WEIGHTS"


ls -1 *.binlog | while read line; do python scripts/tdiff.py -b float $line darknet/$line; done
#ls -1 *.txtlog | while read line; do python scripts/tdiff.py $line darknet/$line; done
