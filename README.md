# Real-Time Face Detection and Smile Classification

This project implements a real-time face detection and smile classification system using OpenCV for face detection and MobileNetV2 for smile classification. The application is fully containerized using Docker for straightforward deployment and usage.

## Features
- **Real-time face detection** using Haar Cascade Classifier
- **Smile classification** using a fine-tuned MobileNetV2 deep learning model
- **Multi-threading** optimized for efficient real-time video capture and processing
- **Offline and real-time data augmentation** to enhance model generalization
- **Single entry point (`app.py`)** for ease of use
- **Dockerized environment** for simplified deployment

## Prerequisites
Before setting up the application, ensure you have installed:
- **Docker** (recommended)
- **Python 3.11** (if running locally without Docker)
- **Git** (for cloning the repository)

## Installation

### Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/cv_assignment.git
cd cv_assignment
```

## Running the Application

### Build Docker Image
To build the Docker image, run:
```bash
docker build -t cv_assignment .
```

### Run Application with Docker
Since Docker is configured to run `app.py` by default, execute the following command:
```bash
docker run -it --rm --device=/dev/video0 cv_assignment
```
- The webcam will automatically activate, detecting and classifying faces in real-time.
- Press **'q'** to exit the application.

### Run Docker Container with Interactive Shell
If needed, you can start an interactive shell in the container without automatically running the application:
```bash
docker run -it --rm cv_assignment /bin/bash
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
2. **Face Detection**: For each frame, the Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) is used to detect faces.
3. **Draw Bounding Box**: Once a face is detected, a rectangle is drawn around it using `cv2.rectangle()`.
4. **Live Tracking**: The script continuously updates the bounding boxes as faces move within the frame.
5. **Dataset Management**: The script allows capturing and splitting datasets for training, data augmentation, and model evaluation.
6. **Model Training and Fine-Tuning**: The user can train the model using labeled data and fine-tune it for better accuracy.
7. **Model Evaluation**: Performance is evaluated using accuracy, precision, recall, and confusion matrix visualization.
8. **Real-Time Detection and Classification**: Once trained, When a face is detected, the trained MobileNetV2 model classifies whether the person is smiling in real-time through a live webcam feed.


## Project Structure
```
cv_assignment/
├── data/                   # Train, validation, and test datasets
├── dataset/                # Original collected images
├── model/                  # Trained and fine-tuned model files
├── report/                 # Final report documentation file
├── src                     # Python scripts
├── app.py                  # Main script
├── Dockerfile              # Docker container setup
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Technologies Used
- **Python**
- **OpenCV**
- **TensorFlow/Keras**
- **Docker**
- **Multithreading**
- **Git**

## Author
- **Nima Daryabar**
- Developed as part of the Junior CV Engineer Technical Assignment at Istituto Italiano di Tecnologia (IIT)

## License
This project is currently private and does not have a license yet.
