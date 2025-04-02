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
    print("1. Face detector (OpenCV Haar Cascade Face Detector)")
    print("2. Face detector (OpenCV DNN + ResNet-Based Face Detector)")
    print("3. Person detector (MobileNetSSD)")
    print("4. Object detector (MobileNetSSD)")
    print("5. Face Detection (ResNet/Multi-processing)")
    print("6. Capture and Split Dataset (Multi-threading)")
    print("7. Capture and Split Dataset (Multi-Multi-processing)")
    print("8. Save Augmented Images")
    print("9. Train Model")
    print("10. Fine-tune Model")
    print("11. Evaluate Model")
    print("12. Real-time Face Detection and Smile Classification")
    print("13. Exit")