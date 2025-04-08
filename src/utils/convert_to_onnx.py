"""
Convert a TensorFlow Keras model to ONNX format.
This script uses the tf2onnx library to perform the conversion.
"""

import os

from pathlib import Path
import tensorflow as tf
import tf2onnx


# --- Paths ---
MODEL_DIR = Path("model/h5_models")
ONNX_OUTPUT_PATH = Path("model/onnx_models")


def get_h5_model_path():
    """
    Get the path of the h5 model from user input.
    The model should be located in the MODEL_DIR directory.
    If the model does not exist, raise a FileNotFoundError.
    
    Returns:
         h5_model_path (Path): The full path of the h5 model.
    """
    h5_model_name = str(input("Enter the h5 model name (without extension): "))
    h5_model_name = f"{h5_model_name}.h5"
    h5_model_path = Path(MODEL_DIR / h5_model_name)
    if not h5_model_path.exists():
        raise FileNotFoundError(f"[Error] Model file {h5_model_path} does not exist.")
    return h5_model_path


def set_onnx_model_path():
    """
    Get the path of the ONNX model from user input.
    The model will be saved in the ONNX_OUTPUT_PATH directory.
    If the directory does not exist, it will be created.
    
     Returns:
        onnx_model_path (Path): The full path of the ONNX model.
    """
    onnx_model_name = str(input("Enter the onnx model name: "))
    onnx_model_path = Path(ONNX_OUTPUT_PATH / onnx_model_name)
    onnx_model_path.mkdir(parents=True, exist_ok=True)
    return onnx_model_path
    

def convert_to_onnx():
    """
    Convert the Keras h5 model to ONNX format.
    The conversion is done using the tf2onnx library.
    """
    h5_model_path = get_h5_model_path()
    onnx_output_path = set_onnx_model_path()
    
    # Load Keras h5 model
    model = tf.keras.models.load_model(h5_model_path)

    # Try to infer input shape
    try:
        input_shape = model.inputs[0].shape  # KerasTensor
        input_signature = [tf.TensorSpec(input_shape, tf.float32)]
    except Exception as e:
        raise RuntimeError(
            "[Error] Could not determine input shape from model. "
            "Ensure your model is built and has a defined input shape."
        ) from e

    # Convert Keras model to ONNX
    onnx_model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

    # Save the ONNX model
    tf2onnx.utils.save_model(onnx_output_path / ".onnx", onnx_model_proto)
    print(f"[Info] Model converted to ONNX format and saved at {onnx_output_path / 'model.onnx'}.")



if __name__ == "__main__":
    convert_to_onnx()