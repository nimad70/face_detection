"""
Gradio UI for Emotion Detection
This script sets up a Gradio interface for the emotion detection model.
"""

import cv2
import gradio as gr
import numpy as np

from src.object_detection.emotion_detection import run_emotion_detection


def process_image(image):
    # Convert PIL to OpenCV format
    image  = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Run emotion detection
    result_img = run_emotion_detection(image)
    # COnvert back to PIL-friendly format
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    return result_img


iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(sources="webcam", type="pil", tool="editor"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="Confidence Threshold"),
        gr.Checkbox(value=True, label="Show Bounding Boxes"),
    ],
    outputs="image",
    title="Real-Time Emotion Detection",
    description="Capture an image from your webcam to detect emotions in images using a TFLite model.",
    live=True,
)

if __name__ == "__main__":
    # Launch the Gradio interface
    iface.launch()