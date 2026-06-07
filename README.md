# AI-Generated Image Detection Using Neural Networks and Digital Forensics

This repository contains the official source code, training pipelines, and evaluation scripts developed for my Diploma Thesis titled **"AI-Generated Image Detection Using Neural Networks"** at the University of Patras (Department of Computer Engineering and Informatics).

The project implements a comprehensive benchmarking suite consisting of **seven (7) distinct neural network configurations** to detect synthetic images (CIFAKE dataset), merging traditional digital forensics with state-of-the-art deep learning.

## Project Overview

The rapid advancement of Generative AI (e.g., Stable Diffusion, Midjourney) makes the generation of photorealistic synthetic images trivial, posing severe threats to digital trust. Traditional deep learning approaches often act as "black boxes" focusing solely on spatial semantic features. 

This project tackles the limitation by combining:
1. **Spatial Domain Analysis:** Standard RGB image analysis via Convolutional Neural Networks (CNNs).
2. **Frequency/Compression Domain Analysis:** **Error Level Analysis (ELA)** to capture anomalous JPEG compression patterns inherent to synthesized images.
3. **Handcrafted Feature Engineering:** **RYB (Red-Yellow-Blue) Color Space** statistical extraction to catch unnatural color distributions that deep neural networks might overlook.

## Tested Architectures & Core Scripts

The repository is structured around standardized training (`train_*.py`) and evaluation (`test_*.py`) scripts for each benchmarked architecture:

### 1. Baseline Feed-Forward Neural Networks (FFNN)
* **`train_model.py` / `test_model.py`**: Trains and evaluates the baseline FFNN architectures using tabular statistical features.
* **`my_first_ffnn_best_model.pth`**: Saved weights for the baseline FFNN.

### 2. Custom Spatial CNN (128x128 RGB Input)
* **`train_cnn128.py` / `test_cnn128.py`**: Handles training and testing for a custom single-stream CNN analyzing raw spatial RGB data.
* **`cnn128_best_model.pth`**: Best performing weights for the spatial CNN stream.

### 3. Custom Forensic ELA CNN
* **`train_cnn_ela.py` / `test_cnn_ela.py`**: Trains a custom CNN purely on computed Error Level Analysis (ELA) structural maps to isolate compression discrepancies.
* **`cnn128_ela_best_model.pth`**: Saved weights for the standalone ELA convolutional network.

### 4. SOTA Transfer Learning via ELA (VGG16 & ResNet18)
* **`train_vgg16_ela.py` / `test_vgg16_ela.py`**: Adapts pre-trained **VGG16** on ELA data to evaluate how deeper deep-learning backbones ingest compression artifacts.
* **`train_resnet_ela.py` / `test_resnet_ela.py`**: Adapts pre-trained **ResNet18** on ELA data.
* **`vgg16_ela_best_model.pth` / `resnet18_ela_best_model.pth`**: Model checkpoints for the respective SOTA networks.

### 5. Multi-Stream & Hybrid Architectures (Proposed Models)
* **`train_Hybrid.py` / `test_Hybrid.py`** and **`train_cnn_rgb_ela.py` / `test_cnn_rgb_ela.py`**: Core experimental scripts orchestrating the multi-branch fusion systems, combining the spatial RGB stream, forensic ELA stream, and handcrafted statistical features before final classification.
* **`hybrid_best_model.pth` / `multi_stream_best_model.pth`**: Checkpoints containing optimized weights for the fusion pipelines.

### 6. Feature Extraction & Engineering
* **`extract_features.py` / `exract_features_test.py`**: Standalone helper scripts to handle preprocessing and compile metadata/color statistics into tabular format (`features.csv`, `features_test.csv`).

## Repository Structure

```Thesis
├── Cifake_Dataset/           # Data directory (ignored by git, must be downloaded separately)
├── *best_model.pth           # Saved model checkpoints for all 7 configurations
├── *plot.png                 # Performance and training loss/accuracy curves
├── extract_features*.py      # Feature engineering and preprocessing utilities
├── *.csv                     # Tabular statistical data arrays
├── scaler*.pkl               # Saved normalization scalers for evaluation inference
├── train_*.py                # Standalone training orchestration scripts
├── test_*.py                 # Standalone validation and evaluation scripts
└── README.md                 # Project documentation