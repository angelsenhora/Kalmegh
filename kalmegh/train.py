"""
Training script for Plant Disease Detection (Kalmegh).
Trains SVM, KNN, Decision Tree models and compares accuracy.
"""

import os
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

from feature_extractor import extract_features

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data/train"
MODEL_DIR  = "models"
LABELS     = {"healthy": 0, "diseased": 1}
LABEL_NAMES = ["Healthy", "Diseased"]
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(data_dir: str):
    X, y = [], []
    for label_name, label_id in LABELS.items():
        folder = os.path.join(data_dir, label_name)
        images = glob.glob(os.path.join(folder, "*.jpg")) + \
                 glob.glob(os.path.join(folder, "*.jpeg")) + \
                 glob.glob(os.path.join(folder, "*.png"))
        print(f"  [{label_name}] {len(images)} images found")
        for img_path in images:
            try:
                features = extract_features(img_path)
                X.append(features)
                y.append(label_id)
            except Exception as e:
                print(f"  Skipping {img_path}: {e}")
    return np.array(X), np.array(y)


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results: dict):
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, [a * 100 for a in accs], color=['#4C72B0', '#DD8452', '#55A868'])
    plt.ylim(0, 110)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, f"{acc*100:.1f}%", ha='center')
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "accuracy_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def _generate_demo_images(data_dir: str, n: int = 40):
    """
    Generate synthetic leaf-like images so the pipeline runs end-to-end
    even without a real dataset. Replace with real images for production.
    """
    import cv2 as _cv2
    rng = np.random.default_rng(42)

    for label in ("healthy", "diseased"):
        folder = os.path.join(data_dir, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(n):
            # Base green canvas
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[:, :, 1] = rng.integers(80, 160)   # green channel

            if label == "diseased":
                # Add brown/yellow spots to simulate disease
                for _ in range(rng.integers(3, 8)):
                    cx, cy = rng.integers(20, 180, size=2)
                    r = rng.integers(8, 25)
                    color = (int(rng.integers(20, 60)),
                             int(rng.integers(60, 120)),
                             int(rng.integers(100, 180)))
                    _cv2.circle(img, (cx, cy), r, color, -1)
            else:
                # Uniform healthy green with slight variation
                img[:, :, 1] += rng.integers(0, 30, (200, 200), dtype=np.uint8)

            # Draw leaf ellipse outline
            _cv2.ellipse(img, (100, 100), (70, 90), 0, 0, 360, (0, 200, 0), 2)
            path = os.path.join(folder, f"demo_{label}_{i:03d}.jpg")
            _cv2.imwrite(path, img)

    print(f"  Generated {n} demo images per class in {data_dir}/")


def train():
    print("\n=== Loading dataset ===")
    X, y = load_dataset(DATA_DIR)

    if len(X) < 10:
        print("\n[WARNING] Very few real images found.")
        print("Generating synthetic demo images for a full end-to-end run...")
        _generate_demo_images(DATA_DIR)
        X, y = load_dataset(DATA_DIR)

    print(f"Total samples: {len(X)}, Features: {X.shape[1] if X.ndim > 1 else 'N/A'}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    models = {
        "SVM":           SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }

    results = {}
    print("\n=== Training models ===")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5)

        print(f"  Test Accuracy : {acc*100:.2f}%")
        print(f"  CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, f"{name} Confusion Matrix",
            os.path.join(MODEL_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
        )

        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))
        results[name] = {'accuracy': acc, 'cv_mean': cv_scores.mean()}

    plot_accuracy_comparison(results)

    # Save best model reference
    best = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n=== Best Model: {best} ({results[best]['accuracy']*100:.2f}%) ===")
    with open(os.path.join(MODEL_DIR, "best_model.txt"), "w") as f:
        f.write(best.lower().replace(' ', '_'))

    print("\nAll models saved to models/")


if __name__ == "__main__":
    train()
