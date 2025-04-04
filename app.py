"""
Main Application Script for Webcam-Based Face Detection and Dataset Creation

This script integrates functionalities from various modules to capture webcam video, detect faces,
create labeled datasets, apply data augmentation, train and fine-tune a model, evaluate model performance, 
and perform real-time face detection and smile classification.
"""

from src.utils.menu import show_menu
from src.object_detection.haar_cascade_face_detector import haar_cascade_face_detector
from src.object_detection.resNet_face_detector import run_face_detection
from src.object_detection.mobileNet_object_detector import run_object_detection
from src.object_detection.customized_face_detection import customized_face_detection
from src.multi_processing.multi_processing_main import multi_processing_main
from src.multi_processing.threaded_main import thread_main
from src.multi_processing.face_detection_multi_threading import start_threads
from src.dataset.dataset_creation import split_dataset, LABELS
from src.dataset.data_augmentation import save_augmented_images
from src.model_training.model_training import train_model, fine_tune_model
from src.model_training.model_evaluation import display_accuracy_metrics, plot_confusuion_matrix


if __name__ == "__main__":
    while True:
        show_menu()
        choice = input("Enter your choice (1-7): ")

        if choice == "1":
            """
            Captures images, splits them into training, validation, and testing datasets.
            Ensures that a minimum of 6 images per category are taken before proceeding.
            """
            thread_main()

            is_splited = True
            while is_splited:
                start_threads()
                res = split_dataset(LABELS) # Split the dataset
                if res:
                    print("Smile and Non-smile datasets are created.")
                    break
                warning_text = "*WARNING: Please take at least 6 shots per each category to add to the dataset!"
                print('-' * len(warning_text))
                print(f"{warning_text}")
                print('-' * len(warning_text))
        
        elif choice == "2":
            """
            Applies data augmentation techniques to the training dataset to enhance generalization.
            """
            print("\n => If you want to augment the dataset, please make sure that you have already created the dataset,")
            print(" => By choosing this option, data augmentation inside model training will be skipped.")
            is_augmented = False
            is_augmented = save_augmented_images()
            print("Augmented images are saved to the corresponding training datasets!")

        elif choice == "3":
            """
            Trains the model using the prepared dataset.
            """
            train_model(is_augmented)
            print("Model training is completed!")
        
        elif choice == "4":
            """
            Fine-tunes the pre-trained model to improve performance.
            """
            fine_tune_model()
            print("Model fine-tuning is completed!")

        elif choice == "5":
            """
            Evaluates the trained or fine-tuned model using accuracy, precision, recall, 
            F1-score, and confusion matrix.
            """
            display_accuracy_metrics(is_fine_tuned=True)
            plot_confusuion_matrix()
            print("Model evaluation is completed!")
        
        elif choice == "6":
            """
            Runs real-time face detection and smile classification using a webcam feed.
            """
            customized_face_detection()

        elif choice == "7":
            """
            Exits the application.
            """
            print("Thank you!")
            print("Exiting the application...")
            break
        
        else:
            """
            Handles invalid menu selections.
            """
            invalid_option_text = "<Invalid choice> --> Please select a valid option between 1 to 7!"
            print('-' * len(invalid_option_text))
            print(f"{invalid_option_text}")
            print('-' * len(invalid_option_text))
