"""
Prediction script for Plant Disease Detection (Kalmegh).
Supports SVM, KNN, Decision Tree (individually) and CNN.
"""

import os
import sys
import joblib
import numpy as np
from feature_extractor import extract_features

MODEL_DIR   = "models"
LABEL_NAMES = {0: "Healthy", 1: "Diseased"}

# Human-readable model names
MODEL_DISPLAY = {
    "svm":           "SVM (RBF Kernel)",
    "knn":           "KNN (k=5)",
    "decision_tree": "Decision Tree",
    "cnn":           "CNN (Deep Learning)",
}


def _load_model(model_name: str):
    """Load a specific .pkl model and the shared scaler."""
    model_path  = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model '{model_name}.pkl' not found. Run train.py first."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "scaler.pkl not found. Run train.py first."
        )

    return joblib.load(model_path), joblib.load(scaler_path)


def predict_ml_by_name(image_path: str, model_name: str) -> dict:
    """Predict using a specific named ML model (svm / knn / decision_tree)."""
    model, scaler = _load_model(model_name)

    features        = extract_features(image_path)
    features_scaled = scaler.transform([features])

    pred  = int(model.predict(features_scaled)[0])
    label = LABEL_NAMES[pred]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features_scaled)[0]
        confidence = round(float(proba[pred]) * 100, 2)

    return {
        "model":      MODEL_DISPLAY.get(model_name, model_name.upper()),
        "prediction": label,
        "confidence": confidence,
        "error":      None,
    }


def predict_best_ml(image_path: str) -> dict:
    """Predict using whichever model was best during training."""
    best_file = os.path.join(MODEL_DIR, "best_model.txt")
    model_name = "svm"
    if os.path.exists(best_file):
        with open(best_file) as f:
            model_name = f.read().strip()
    return predict_ml_by_name(image_path, model_name)


def predict_cnn(image_path: str) -> dict:
    """Predict using the trained CNN model (requires TensorFlow)."""
    try:
        import tensorflow as tf
        import cv2
    except ImportError as e:
        return {
            "model":      "CNN",
            "prediction": "Unknown",
            "confidence": None,
            "error":      f"TensorFlow not installed: {e}",
        }

    cnn_path = os.path.join(MODEL_DIR, "cnn_model.keras")
    if not os.path.exists(cnn_path):
        return {
            "model":      "CNN",
            "prediction": "Unknown",
            "confidence": None,
            "error":      "CNN model not found. Run train_cnn.py first.",
        }

    model = tf.keras.models.load_model(cnn_path)
    img   = cv2.imread(image_path)
    if img is None:
        return {
            "model":      "CNN",
            "prediction": "Unknown",
            "confidence": None,
            "error":      f"Cannot read image: {image_path}",
        }

    img  = cv2.resize(img, (128, 128)).astype(np.float32) / 255.0
    img  = np.expand_dims(img, axis=0)
    prob = float(model.predict(img, verbose=0)[0][0])
    pred = 1 if prob >= 0.5 else 0

    return {
        "model":      "CNN (Deep Learning)",
        "prediction": LABEL_NAMES[pred],
        "confidence": round((prob if pred == 1 else 1.0 - prob) * 100, 2),
        "error":      None,
    }


def predict(image_path: str, use_cnn: bool = False) -> dict:
    """Main entry point used by app.py."""
    if use_cnn:
        return predict_cnn(image_path)
    return predict_best_ml(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [svm|knn|decision_tree|cnn]")
        sys.exit(1)

    img_path   = sys.argv[1]
    model_arg  = sys.argv[2] if len(sys.argv) > 2 else "svm"

    if model_arg == "cnn":
        result = predict_cnn(img_path)
    else:
        result = predict_ml_by_name(img_path, model_arg)

    print(f"\nModel      : {result['model']}")
    if result.get("error"):
        print(f"Error      : {result['error']}")
    else:
        print(f"Prediction : {result['prediction']}")
        if result["confidence"] is not None:
            print(f"Confidence : {result['confidence']:.2f}%")
