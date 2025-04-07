"""
Main Application Script for Webcam-Based Face Detection and Dataset Creation

This script integrates functionalities from various modules to capture webcam video, detect faces,
create labeled datasets, apply data augmentation, train and fine-tune a model, evaluate model performance, 
and perform real-time face detection and smile classification.
"""

import tensorflow as tf
from tensorflow import keras

from src.data_pipeline.augment_data_pipeline import save_augmented_images
from src.data_pipeline.tarining_data_pipeline import split_dataset, LABELS
from src.model_training.model_training_EfficientNetB0 import (
    fine_tune_model, 
    get_user_input, 
    save_history_plot,
    train_model, 
    train_with_kfold,
    evaluate_and_visualize_misclassifications,
    MODEL_PATH
)
from src.multi_processing.threaded_data_pipeline_resNet import data_pipeline_thread
from src.object_detection.customized_face_detection import customized_face_detection
from src.utils.menu import show_menu


if __name__ == "__main__":
    # Initialize variables
    is_augmented = False

    while True:
        show_menu()
        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            """
            Captures images, splits them into training, validation, and testing datasets.
            Ensures that a minimum of 6 images per category are taken before proceeding.
            """
            is_splited = True
            while is_splited:
                data_created = data_pipeline_thread()
                if data_created:
                    res = split_dataset(LABELS) # Split the dataset
                    if res:
                        print("[INFO] Smile and Non-smile datasets are created.")
                        break
                    warning_text = "[WARNING] Please take at least 6 shots per each category to add to the dataset!"
                    print('-' * len(warning_text))
                    print(f"{warning_text}")
                    print('-' * len(warning_text))
                    try:
                        exit_request = input("\nDo you want to exit? (y/n): ").strip().lower()
                        if exit_request == 'y':
                            print("Exiting the dataset creation...")
                            is_splited = False
                            break
                        elif exit_request == 'n':
                            print("Continuing to take more shots...")
                    except ValueError:
                        print("[WARNING] Invalid input. Please enter 'y' or 'n'.")
                else:
                    print("Exiting...")
                    break
        
        elif choice == "2":
            """
            Applies data augmentation techniques to the training dataset to enhance generalization.
            """
            user_consent = input("\nDo you want to skip data augmentation inside model training? (y/n): ").strip().lower()
            
            if user_consent == 'y':
                while True:
                    is_augmented = save_augmented_images()
                    if not is_augmented:
                        print("\n[WARNING] If you want to augment the dataset, please make sure you already have created the dataset.")
                        break
                    elif is_augmented:
                        continue_aug = input("\nDo you want to continue data augmentation? (y/n): ").strip().lower()
                        if continue_aug == 'y':
                            print("\n[INFO] Data augmentation is performed one more time.")
                        elif continue_aug == 'n':
                            break
                        else:
                            print("[WARNING] Invalid input. Please enter 'y' or 'n'.")
                            continue
            elif user_consent == 'n':
                print("\n[INFO] Data augmentation is skipped.")
                print("[INFO] It is included in model training.")
            
            else:
                print("[WARNING] Invalid input. Defaulting to skipping data augmentation.")

        elif choice == "3":
            """
            Trains the CNN model using the training dataset. The user is prompted to choose
            whether to use a pre-split dataset or perform k-fold cross-validation.
            Evaluates the model's performance and displays accuracy metrics.
            """
            use_pre_split = get_user_input(is_application=True)
            if use_pre_split:
                history = train_model(augmented=is_augmented)
                save_history_plot(history, "presplit")
                fine_tune_model()
                model = tf.keras.models.load_model(MODEL_PATH / "smile_detection_fine_tuned_model.h5")
            else:
                train_with_kfold(augmented=is_augmented)
                model = tf.keras.models.load_model(MODEL_PATH / "smile_model_fold1_acc_" + max([f.name for f in MODEL_PATH.glob("smile_model_fold*_acc_*.h5")], key=lambda f: float(f.stem.split("_acc_")[-1])).split("/")[-1])
            
            evaluate_and_visualize_misclassifications(model)
        
        elif choice == "5":
            """
            Runs real-time face detection and smile classification using a webcam feed.
            """
            customized_face_detection()

        elif choice == "6":
            """
            Exits the application.
            """
            print("\nThank you!")
            print("Exiting the application...")
            break
        
        else:
            """
            Handles invalid menu selections.
            """
            invalid_option_text = "<Invalid choice> --> Please select a valid option between 1 to 6!"
            print('-' * len(invalid_option_text))
            print(f"{invalid_option_text}")
            print('-' * len(invalid_option_text))
