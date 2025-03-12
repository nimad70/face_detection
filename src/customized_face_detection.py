# new_face_detection.py
import cv2
import tensorflow as tf
import numpy as np
from face_detection import video_capture, haarcascade_classifier, preprocessing_frame, face_detector, scale_bounding_box
from model_evaluation import load_model

# Hyperparameters
IMG_SIZE = 224


def customized_face_detection():
    is_fine_tuned_model = True
    model = load_model(is_fine_tuned_model)

    cap = video_capture()
    face_classifier = haarcascade_classifier()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        processed_frame = preprocessing_frame(frame)
        detected_faces = face_detector(face_classifier, processed_frame)
        scaled_faces = scale_bounding_box(frame, detected_faces)

        for (x, y, w, h) in scaled_faces:
            face_roi = frame[y:y+h, x:x+w] # Region of interest
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            
            # Add batch dimension for model prediction (300, 300, 3) -> (1, 300, 300, 3) 
            face_expanded = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_expanded) # Make prediction
            
        cv2.imshow("Smile Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    customized_face_detection()