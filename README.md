# Face-Mask-Detect
from pathlib import Path

readme = """# Face Mask Detection using Deep Learning

A computer vision project that classifies images into **With Mask** and **Without Mask** categories. It uses transfer learning with the pre-trained **VGG16** model and includes an OpenCV webcam loop for live prediction.

## Project Overview

The model is trained on face images organized into two classes:

- `with_mask` — faces wearing a mask
- `without_mask` — faces not wearing a mask

The notebook loads and resizes images to **224 × 224**, normalizes pixel values, splits the dataset into training and testing sets, and trains a binary classifier using VGG16 features.

## Tech Stack

- Python
- TensorFlow / Keras
- VGG16 (transfer learning)
- OpenCV
- NumPy
- scikit-learn
- Jupyter Notebook

## Repository Structure

```text
FACE-MASK-DETECTION/
├── face_mask_detector.ipynb
├── frontal-face-detection.ipynb
├── Untitled.ipynb
├── train/
│   ├── with_mask/
│   └── without_mask/
└── sample/
    ├── 1.jpg
    └── ...
