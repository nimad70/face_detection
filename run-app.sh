#!/bin/bash

# Check if the camera is connected
check_camera() {
    if v412-ctl --list-devices &>/dev/null; then
        return 0 # Camera found
    else
        return 1 # No camera detected
}


# Run the application if the camera is available
if check_camera; then
    echo "Camera is connected. Launching the application..."
    docker run -it --rm --device=/dev/video0 cv_assignment
else
    echo "Warning: No camera detected! Please connect a camera and try again."
    exit 1
fi