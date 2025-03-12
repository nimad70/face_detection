"""
Main Application Script for Webcam-Based Face Detection and Dataset Creation

This script integrates functionalities from various modules to capture webcam video, detect faces,
create labeled datasets, and split data into training, validation, and testing sets.
"""

import cv2
import sys
from src.face_detection import video_capture, haarcascade_classifier, face_detector, draw_rectangle, preprocessing_frame
from src.face_detection_multi_threading import start_threads
from src.dataset_creation import split_dataset, LABELS
from src.data_augmentation import save_augmented_images
from src.model_training import train_model, fine_tune_model
from src.model_evaluation import display_accuracy_metrics, plot_confusuion_matrix
from src.customized_face_detection import customized_face_detection


def main():
    """
    Captures video from the webcam, detects faces, and displays the processed frames with bounding boxes from 
    face_detection.py module.
    
    The function continuously reads frames from the webcam, preprocesses them, detects faces using Haar Cascade classifier,
    and draws bounding boxes around detected faces. The video stream is displayed in a window, and the loop exits
    when the user presses 'q'.
    """
    cap = video_capture()
    classifier = haarcascade_classifier()

    while True:
        res, frame = cap.read() # capture frames from the video source

        # If the frame is not captured, then break the loop
        if not res:
            print("Failed to capture frames")
            break

        preprocessed_frame = preprocessing_frame(frame)
        face_classifier = haarcascade_classifier()
        detected_faces = face_detector(face_classifier, preprocessed_frame)
        draw_rectangle(frame, detected_faces)

        cv2.imshow("Webcam", frame) # Display the captured frame

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


def show_menu():
    print("\nPlease select an option from the menu:")
    print("1. Capture face")
    print("2. Capture and Split Dataset")
    print("3. Save Augmented Images")
    print("4. Train Model")
    print("5. Fine-tune Model")
    print("6. Evaluate Model")
    print("7. Real-time Face Detection and Smile Classification")
    print("8. Exit")


if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("Enter your choice (1-8): ")

        if choice == "1":
            main()
        
        elif choice == "2":
            is_splited = True
            while is_splited:
                start_threads()
                res = split_dataset(LABELS) # Split the dataset
                if res:
                    print("Smile and Non-smile datasets are splited")
                    break
                warning_text = "*WARNING: Please take at least 6 shots per each category to add to the dataset"
                print('-' * len(warning_text))
                print(f"{warning_text}")
                print('-' * len(warning_text))
        
        elif choice == "3":
            save_augmented_images()
            print("Augmented images are saved to the corresponding training datasets")

        elif choice == "4":
            # Train the model
            train_model()
            print("Model training is completed")
        
        elif choice == "5":
            # Fine-tune the model
            fine_tune_model()
            print("Model fine-tuning is completed")

        elif choice == "6":
            # Evaluate the model
            display_accuracy_metrics(is_fine_tuned=True)
            plot_confusuion_matrix()
            print("Model evaluation is completed")
        
        elif choice == "7":
            # Perform real-time face detection and smile classification
            customized_face_detection()

        elif choice == "8":
            print("Exiting the application!")
            break
        
        else:
            invalid_option_text = "Invalid choice. Please select a valid option between 1 to 8."
            print('-' * len(invalid_option_text))
            print(f"{invalid_option_text}")
            print('-' * len(invalid_option_text))
