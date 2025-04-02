"""
Model Evaluation Script

This script evaluates a trained and fine-tuned models by loading test data,
making predictions, computing evaluation metrics, and displaying the confusion matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Constants
DATA_PATH = Path("data")
MODEL_PATH = Path("model")

# Load test dataset
TEST_DATA_PATH = Path("data/test")

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32


def load_test_data():
    """
    Loads the test dataset from the test directory.

    Returns:
        test_ds (tf.data.Dataset): The test dataset.
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DATA_PATH),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = test_ds.class_names
    print(f"Test dataset class names: {class_names}")
    
    return test_ds


def load_model(is_fine_tuned_model=True):
    """
    Loads a trained or fine-tuned model from storage.

    Args:
        is_fine_tuned_model (bool, optional): If True, loads the fine-tuned model. Defaults to True.

    Returns:
        model (tf.keras.Model): The loaded model.
    """
    if is_fine_tuned_model:
        # For fine-tuned model evaluation
        model_save_path = MODEL_PATH / "smile_detection_fine_tuned_model.h5"
        model = tf.keras.models.load_model(model_save_path)
    else:
        # Load the trained model
        model_save_path = MODEL_PATH / "smile_detection_model.h5"
        model = tf.keras.models.load_model(model_save_path)
    
    return model


def model_prediction(is_fine_tuned_model=True):
    """
    Makes predictions on the test dataset using the specified model.

    Args:
        is_fine_tuned_model (bool, optional): If True, uses the fine-tuned model. Defaults to True.

    Returns:
        true_labels, predicted_labels (tuple): True labels and predicted labels.
    """
    test_ds = load_test_data()
    model = load_model(is_fine_tuned_model)

    # Ectract the corresponding labels
    true_labels = np.concatenate([y.numpy() for _, y in test_ds])

    # Generate predictions for test images
    predictions = model.predict(test_ds)

    # Convert predcitions (probabilities) into binary labels (0 and 1) using 0.5 threshold
    predicted_labels = (predictions.flatten() > 0.5).astype(int)

    return true_labels, predicted_labels


def evaluate_model(is_fine_tuned=False):
    """
    Evaluates the model using accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        is_fine_tuned (bool, optional): If True, evaluates the fine-tuned model. Defaults to False.

    Returns:
        accuracy, precision, recall, f1, conf_matrix (tuple): Accuracy, precision, recall, F1-score, and confusion matrix.
    """
    true_labels, predicted_labels = model_prediction(is_fine_tuned)

    if is_fine_tuned:
        print("Fine-tuned model evaluation:")
    else:
        print("Initial model evaluation:")
    
    accuracy = accuracy_score(true_labels, predicted_labels) # (TP + TN) / Total Samples
    precision = precision_score(true_labels, predicted_labels) # (TP / (TP + FP)
    recall = recall_score(true_labels, predicted_labels) # (TP / (TP + FN))
    f1 = f1_score(true_labels, predicted_labels) # Harmonic mean of precision and recal
    conf_matrix = confusion_matrix(true_labels, predicted_labels) # Confusion matrix

    return accuracy, precision, recall, f1, conf_matrix


def display_accuracy_metrics(is_fine_tuned=False):
    """
    Displays the accuracy, precision, recall, and F1-score of the model.

    Args:
        is_fine_tuned (bool, optional): If True, evaluates the fine-tuned model. Defaults to False.
    """
    acc_metrics = ()
    acc_metrics = evaluate_model(is_fine_tuned)
    
    # Display evaluation metrics
    print(f"Accuracy: {acc_metrics[0]:.4f}")
    print(f"Precision: {acc_metrics[1]:.4f}")
    print(f"Recall: {acc_metrics[2]:.4f}")
    print(f"F1-Score: {acc_metrics[3]:.4f}")


def plot_confusuion_matrix():
    """
    Plots the confusion matrix for the model evaluation.
    """
    acc_metrics = []
    acc_metrics = evaluate_model()
    
    fig, ax = plt.subplots(figsize=(5, 5))

    class_names = ['Not Smiling', 'Smiling']
    disp = ConfusionMatrixDisplay(confusion_matrix=acc_metrics[4], display_labels=class_names)
    disp.plot(cmap='RdBu', ax=ax)
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    display_accuracy_metrics(is_fine_tuned=True)
    plot_confusuion_matrix()