# 🌿 Kalmegh – Plant Disease Detection from Leaf Images

Kalmegh is a machine learning project that detects whether a plant leaf is **Healthy** or **Diseased** using classical ML models (SVM, KNN, Decision Tree) and a bonus CNN model. It includes a Flask web app for real-time predictions.

---

## Project Structure

```
kalmegh/
├── data/
│   ├── train/
│   │   ├── healthy/        ← training images (healthy leaves)
│   │   └── diseased/       ← training images (diseased leaves)
│   ├── test/
│   │   ├── healthy/
│   │   └── diseased/
│   └── sample/             ← sample images for quick testing
├── models/                 ← saved .pkl and .keras model files
├── notebooks/
│   └── kalmegh_training.ipynb
├── static/
│   ├── css/style.css
│   └── uploads/            ← uploaded images (auto-created)
├── templates/
│   ├── index.html
│   └── result.html
├── feature_extractor.py    ← preprocessing + feature extraction
├── train.py                ← train SVM, KNN, Decision Tree
├── train_cnn.py            ← train CNN (TensorFlow/Keras)
├── predict.py              ← prediction script (CLI)
├── app.py                  ← Flask web application
├── requirements.txt
└── .gitignore
```

---

## Features

- **Preprocessing**: Gaussian blur (noise removal), resize to 128×128, normalization
- **Feature Extraction**:
  - Color: HSV histogram (192 features)
  - Texture: GLCM properties + LBP histogram (25 features)
  - Shape: Contour area, circularity, Hu moments (9 features)
- **Models**: SVM (RBF kernel), KNN (k=5), Decision Tree + cross-validation comparison
- **CNN Bonus**: Lightweight 3-block CNN with BatchNorm, Dropout, EarlyStopping
- **Confidence Score**: Shown for all models that support `predict_proba`
- **Flask Web App**: Upload image → instant prediction with confidence bar

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/kalmegh.git
cd kalmegh
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your dataset

Place leaf images in:
```
data/train/healthy/     ← healthy leaf images (.jpg/.png)
data/train/diseased/    ← diseased leaf images (.jpg/.png)
```

> Recommended: 100+ images per class for good accuracy.
> A great free dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## Usage

### Train ML Models

```bash
python train.py
```

Outputs saved to `models/`: `svm.pkl`, `knn.pkl`, `decision_tree.pkl`, `scaler.pkl`, confusion matrix PNGs, accuracy comparison chart.

### Train CNN (Bonus)

```bash
python train_cnn.py
```

Saves `models/cnn_model.keras` and training curves.

### Predict from CLI

```bash
# Using best ML model
python predict.py path/to/leaf.jpg

# Using CNN
python predict.py path/to/leaf.jpg --cnn
```

### Run Flask Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook kalmegh_training.ipynb
```

---

## Model Performance (example with PlantVillage dataset)

| Model         | Test Accuracy | CV Accuracy |
|---------------|--------------|-------------|
| SVM (RBF)     | ~94%         | ~93%        |
| KNN (k=5)     | ~89%         | ~88%        |
| Decision Tree | ~86%         | ~85%        |
| CNN           | ~97%         | –           |

> Actual results depend on your dataset size and quality.

---

## Tech Stack

- Python 3.10+
- OpenCV – image preprocessing
- NumPy – numerical operations
- Scikit-learn – ML models
- scikit-image – GLCM texture features
- TensorFlow/Keras – CNN model
- Flask – web application
- Matplotlib / Seaborn – visualisations

---

## License

MIT License. Free to use and modify.
