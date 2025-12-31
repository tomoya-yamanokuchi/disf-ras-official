#!/bin/bash
set -e

# ホスト側のユーザ名（例: tomoya-y）
USER=${USER:-$(whoami)}
# DEVICE_ARGS=""



docker run -it --rm --gpus all --privileged --net=host --ipc=host \
  --device=/dev/video0 --device=/dev/video1 --device=/dev/video2 --device=/dev/video3 \
  --device=/dev/video4 --device=/dev/video5 --device=/dev/video6 --device=/dev/video7 \
  --group-add 44 \
  -e DISPLAY=$DISPLAY \
  -e AUDIODEV="hw:Device, 0" \
  -e XAUTHORITY=/home/$(id -un)/.Xauthority \
  -v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/$USER/disf-ras-official:/home/cudagl/disf_ras \
  -v /home/$USER/data:/home/cudagl/data \
  -v /home/$USER/ur3e_grasp:/home/cudagl/disf_ras/ur3e_grasp \
  -v /home/$USER/webcam_control:/home/cudagl/disf_ras/webcam_control \
  -w /home/cudagl/disf_ras \
  docker_disf_ras:latest \
  bash
