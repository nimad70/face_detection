import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Constants
DATA_PATH = Path("data")
IMG_SIZE = 224
BATCH_SIZE = 32

# Load test dataset
TEST_DATA_PATH = Path("data/test")


def load_test_data():
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
    if is_fine_tuned_model:
        # For fine-tuned model evaluation
        model = tf.keras.models.load_model("smile_detection_fine_tuned_model.h5")
    else:
        # Load the trained model
        model = tf.keras.models.load_model("smile_detection_model.h5")
    
    return model


# Obtain true labels and predictions
def model_prediction(is_fine_tuned_model=True):
    test_ds = load_test_data()
    model = load_model(is_fine_tuned_model)

    # Ectract the corresponding labels
    true_labels = np.concatenate([y.numpy() for _, y in test_ds])

    # Generate predictions for test images
    predictions = model.predict(test_ds)

    # Convert predcitions (probabilities) into binary labels (0 and 1) using 0.5 threshold
    predicted_labels = (predictions.flatten() > 0.5).astype(int)

    return true_labels, predicted_labels


# Calculate evaluation metrics
def evaluate_model():
    # is_fine_tuned = True
    is_fine_tuned = False

    # Retrieve true lables and predictions
    true_labels, predicted_labels = model_prediction(is_fine_tuned)

    if is_fine_tuned:
        print("Fine-tuned model evaluation:")
    else:
        print("Initial model evaluation:")
    
    # (TP + TN) / Total Samples
    accuracy = accuracy_score(true_labels, predicted_labels)

    # model soundness: (TP / (TP + FP)
    precision = precision_score(true_labels, predicted_labels)

    # model completeness: (TP / (TP + FN))
    recall = recall_score(true_labels, predicted_labels)

    # Harmonic mean of precision and recal
    f1 = f1_score(true_labels, predicted_labels)

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)


    return accuracy, precision, recall, f1, conf_matrix


if __name__ == "__main__":
    acc_metrics = ()
    acc_metrics = evaluate_model(is_fine_tuned=False)

    # Display evaluation metrics
    print(f"Accuracy: {acc_metrics[0]:.4f}")
    print(f"Precision: {acc_metrics[1]:.4f}")
    print(f"Recall: {acc_metrics[2]:.4f}")
    print(f"F1-Score: {acc_metrics[3]:.4f}")
    