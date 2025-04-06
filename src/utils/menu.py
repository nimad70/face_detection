"""
This module provides a menu interface for the user to interact with the face recognition system.
It allows the user to select various functionalities such as capturing a face, dataset creation,
data augmentation, model training, fine-tuning, evaluation, and real-time face detection.
"""


def show_menu():
    """
    Displays the main menu options for the user.

    The user can select various functionalities such as capturing a face, 
    dataset creation, data augmentation, model training, fine-tuning, evaluation, 
    and real-time face detection.
    """
    print("\nPlease select an option from the menu:")
    print("1. Create a dataset")
    print("2. Augment the dataset")
    print("3. Train the CNN model")
    print("4. Fine-tune the model")
    print("5. Evaluate model")
    print("6. Smile detection")
    print("7. Exit")