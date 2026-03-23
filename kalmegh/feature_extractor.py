"""
Feature extraction module for plant disease detection.
Extracts color, texture, and shape features from leaf images.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


IMG_SIZE = (128, 128)


def preprocess_image(image_path: str) -> np.ndarray:
    """Load, denoise, resize, and normalize an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Noise removal with Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Resize to fixed dimensions
    img = cv2.resize(img, IMG_SIZE)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


def extract_color_features(img: np.ndarray) -> np.ndarray:
    """Extract HSV color histogram features (64 bins per channel)."""
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)

    features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features)


def extract_texture_features(img: np.ndarray) -> np.ndarray:
    """Extract GLCM texture features and LBP histogram."""
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    # GLCM features
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        glcm_features.extend(graycoprops(glcm, prop).flatten())

    # LBP histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    return np.concatenate([glcm_features, lbp_hist])


def extract_shape_features(img: np.ndarray) -> np.ndarray:
    """Extract shape features using contours and Hu moments."""
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate leaf
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(9)

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))

    # Hu moments (7 values)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return np.concatenate([[area, circularity], hu])


def extract_features(image_path: str) -> np.ndarray:
    """Full pipeline: preprocess + extract all features."""
    img = preprocess_image(image_path)
    color = extract_color_features(img)
    texture = extract_texture_features(img)
    shape = extract_shape_features(img)
    return np.concatenate([color, texture, shape])
