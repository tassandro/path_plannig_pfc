#!/usr/bin/env bash
set -e
cd /mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc
mkdir -p /tmp/frames
rm -f /tmp/frames/*.png
ffmpeg -y -i output/jackal_dji_demo_teleport.mp4 -vf "select='not(mod(n\,1200))'" -vsync 0 /tmp/frames/f_%03d.png
ls -la /tmp/frames
