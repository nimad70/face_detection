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
    print("1. Capture face")
    print("2. Capture and Split Dataset")
    print("3. Save Augmented Images")
    print("4. Train Model")
    print("5. Fine-tune Model")
    print("6. Evaluate Model")
    print("7. Real-time Face Detection and Smile Classification")
    print("8. Exit")