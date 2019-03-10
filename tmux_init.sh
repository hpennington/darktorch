#!/usr/bin/env bash

# Create session or attach if exists
tmux attach -t darktorch || \
tmux \
	new -d -s darktorch \; \
	rename-window -t 0 root \; \
	send-keys -t darktorch 'source activate darktorch' C-m \; \
	attach -t darktorch