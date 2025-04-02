import numpy as np
from pathlib import Path


DIR_ = Path("model")

def mobileNetSSD_config():
    """
    Configuration for the MobileNet SSD model.
    This function sets the parameters for the MobileNet SSD model including the model directory,
    prototxt file, caffemodel file, confidence threshold, and image size.

    Returns:
        tuple: A tuple containing the model configuration parameters.
    """
    GPU_enabled = True
    model_dir = DIR_ / "MobileNetSSD"
    prototxt = model_dir / "MobileNetSSD_deploy.prototxt"
    caffemodel = model_dir / "MobileNetSSD_deploy.caffemodel"

    # Load labels
    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Detection settings
    confidence_treshold = 0.5
    img_size = (300, 300)

    return (GPU_enabled, prototxt, caffemodel, confidence_treshold, img_size, classes, colors)


def resNetSSD_config():
    """
    Configuration for the ResNet face detection model.
    This function sets the parameters for the ResNet model including the model directory,
    prototxt file, caffemodel file, confidence threshold, and image size.
        
    Returns:
        tuple: A tuple containing the model configuration parameters.
    """
    GPU_enabled = True
    model_dir = DIR_ / "ResNetSSD"
    prototxt = model_dir / "deploy.prototxt"
    caffemodel = model_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

    # Detection settings
    confidence_treshold = 0.5
    img_size = (300, 300)

    return (GPU_enabled, prototxt, caffemodel, confidence_treshold, img_size)


