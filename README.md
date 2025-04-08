# Real-Time Emotion Detection Classification

This project implements a real-time face detection and expression classification system using OpenCV for face detection, EfficientNetV2 models for emotion classification, and deployment via Docker or Streamlit.

## Features

- **Real-time face detection** using ResNet SSD (Caffe-based)
- **Real-time face detection** using Haar Cascade Classifier
- **Real-time Object detection** using MobileNet Classifier
- **Smile classification** using a fine-tuned MobileNetV2 deep learning model
- **Smile & Emotion classification** via EfficientNetB0
- **Offline and real-time data augmentation** to enhance model generalization
- **Multi-threading** optimized for efficient real-time video capture and processing
- **Multiclass emotion support** with bounding box + confidence overlay
- **Dockerized environment** for simplified deployment
- **Single entry point (`app.py`)** for ease of use
- **TFLite** script for high-speed real-time use
- **Gradio UI** for live webcam-based inference
- **Streamlit Web App** for live webcam-based inference
- **Configurable UI:** toggle bounding boxes, set confidence threshold

## Prerequisites

Before setting up the application, ensure you have installed:

### If running locally:

- Python 3.11
- Git

### If running in container:

- Docker installed
- Webcam access enabled

## Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/emotion_detection.git
cd emotion_detection
```

## Running the Application

### Build Docker Image

To build the Docker image, run:

```bash
docker build -t emotion_detection .
```

### Run Application with Docker

Since Docker is configured to run `app_cli.py` by default, execute the following command:

```bash
docker run -it --rm --device=/dev/video0 -p 8501:8501 emotion_detection
```

- The webcam will automatically activate, detecting and classifying faces in real-time.
- Press **'q'** to exit the application.

- Uncomment the corresponing line to run Streamlit interface.
- open http://localhost:8501 in your browser.

### Run Docker Container with Interactive Shell

If needed, you can start an interactive shell in the container without automatically running the application:

```bash
docker run -it --rm emotion_detection /bin/bash
```

### Running Without Docker

If you prefer to run the application without Docker, follow these steps:

#### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run the Application

```bash
python app.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

### Run Gradio UI

```bash
python ./src/deployment/gradio_ui.py
```

### Running with Bash Script

An alternative way to run the application is by using the `runapp.sh` script, which checks for a connected camera before launching the application in Docker.

```bash
./runapp.sh
```

- If a camera is connected, the application launches inside Docker.
- If no camera is detected, a warning message is displayed.

Make sure to give execution permission to the script before running it:

```bash
chmod a+x runapp.sh
```

## Application Functionalities

The `app.py` script serves as the central control point for various functionalities. Different options can be selected by user via a menu.

## How It Works

1. **Capture Video Feed**: The script uses `cv2.VideoCapture()` to access the webcam and retrieve real-time frames.
2. **Face Detection**: For each frame, various face/object detections are used to detect faces.
3. **Draw Bounding Box**: Once a face is detected, a rectangle is drawn around it using `cv2.rectangle()`.
4. **Live Tracking**: The script continuously updates the bounding boxes as faces move within the frame.
5. **Dataset Management**: The script allows capturing and splitting datasets for training, data augmentation, and model evaluation.
6. **Model Training and Fine-Tuning**: The user can train the model using labeled data and fine-tune it for better accuracy.
7. **Model Evaluation**: Performance is evaluated using accuracy, precision, recall, and confusion matrix visualization.
8. **Real-Time Detection and Classification**: Once trained, When a face is detected, the trained MobileNetV2 model classifies whether the person is smiling in real-time through a live webcam feed.
9. **Model Deployment via TFLite**
   The trained model is converted to TensorFlow Lite format and loaded using the tflite.Interpreter for fast, efficient inference on edge devices.

## Project Structure

```
emotion_detection/
├── data/                   # Train, validation, and test datasets
├── dataset/                # Original collected images
├── model/                  # Trained and fine-tuned model files
├── src                     # Python scripts
├── .gitignore              # files and directories to be ignored by Git
├── app_cli.py              # Main script
├── app.py                  # Streamlit Main script
├── Dockerfile              # Docker container setup
├── playground.py           # Run all implemented modules
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
└── runapp.sh               # Bash script to check for camera and launch the app
```

## Technologies Used

- **Python**
- **OpenCV**
- **TensorFlow/Keras**
- **TensorFlow Lite (TFLite)**
- **Docker**
- **Multithreading**
- **Multiprocessing**
- **Gradio UI**
- **Streamlit web app**

## Author

- **Nima Daryabar**

## License

This project is currently private and does not have a license yet.
