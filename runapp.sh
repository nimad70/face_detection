#!/bin/bash

# Check if the camera is connected
if v412-ctl --list-devices;
then
    echo "Camera is connected"
    docker run -it --rm --device=/dev/video0 cv_assignment
else
    echo "Camera is not connected"
fi