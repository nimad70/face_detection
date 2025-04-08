import cv2
import numpy as np
import streamlit as st
import tempfile

from src.object_detection.emotion_detection_gradio import run_emotion_detection


st.set_page_config(page_title="Real-Time Emotion Detection", layout="wide")
st.title("Real-Time Emotion Detection with Streamlit")

threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.01)
show_boxes = st.checkbox("Show Bounding Boxes", value=True)

FRAME_WINDOW = st.image([])

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot access webcam.")
    st.stop()

st.success("Webcam initialized. Press Stop to end.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Frame not read properly.")
        break

    # Pass config to detection function
    frame = run_emotion_detection(frame, confidence_threshold=threshold, show_boxes=show_boxes)

    # Convert BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

    # Exit button
    if st.button("Stop"):
        break

# Cleanup
cap.release()
st.info("Webcam stopped.")
