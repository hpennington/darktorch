#!/usr/bin/env bash

rsync ../darktorch hpenn@$BBPIP:~/Downloads \
    --exclude-from=.gitignore \
    -av

tmux attach -t YOLO \
    ttab ssh hpenn@$BBPIP \
            -L 6006:localhost:6006 \
            -L 8097:localhost:8097 \
            -L 8888:localhost:8888 \
            -t "cd Downloads/darktorch && bash tmux_init.sh" \
    || \
    tmux new -d -s YOLO \; \
        rename-window -t 0 watchman \; \
        send-keys -t YOLO \
            "watchman-make -p '**' --run \
                'rsync -av --exclude-from=.gitignore ../darktorch hpenn@$BBPIP:~/Downloads'
            " C-m \; \
    && \
    ttab ssh hpenn@$BBPIP \
            -L 6006:localhost:6006 \
            -L 8097:localhost:8097 \
            -L 8888:localhost:8888 \
            -t "cd Downloads/darktorch && bash tmux_init.sh"

tmux attach -t YOLO

