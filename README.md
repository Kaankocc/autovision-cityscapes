# 🏎️ AutoVision: Semantic Segmentation for Autonomous Driving (Phase 1)

> **A deep learning pipeline for pixel-level urban scene understanding, fine-tuned on the Cityscapes dataset using a custom UNet architecture.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-In--Progress-orange.svg)]()

---

## 📌 Project Overview

AutoVision is an ongoing research project aimed at mastering **Semantic Segmentation** for self-driving vehicles. This repository documents the evolution of a segmentation model, starting from a baseline architecture and moving toward advanced optimization for real-world driving sequences like those found in Stuttgart.

Phase 1 focused on establishing a robust PyTorch data pipeline, implementing a custom deep-depth UNet, and performing initial fine-tuning to capture critical "agent" classes like pedestrians and vehicles.

### 🚀 Phase 1 Results (mIoU)

The model was evaluated on the **Cityscapes validation set** ($512 \times 512$ resolution) after targeted fine-tuning.

| Category         | Foundation Score | Phase 1 Score | Change     |
| :--------------- | :--------------- | :------------ | :--------- |
| **Road** | 88.29%           | **89.85%** | ✅ +1.56%  |
| **Vehicle** | 66.10%           | **67.26%** | ✅ +1.17%  |
| **Construction** | 67.72%           | **69.46%** | ✅ +1.74%  |
| **Object** | 17.91%           | **20.57%** | ✅ +2.65%  |
| **Nature** | 75.59%           | **76.26%** | ✅ +0.67%  |

---

## 🛠️ Methodology

### 1. Data Processing
- **Cityscapes Integration:** Handled the high-resolution cityscapes image pairs by implementing a "mask baking" script to convert raw labels into 7-class semantic maps.
- **Class Mapping:** Standardized urban elements into 7 key categories: *Background, Road, Human, Vehicle, Construction, Object, and Nature*.

### 2. Model Architecture: Custom Deep UNet
We implemented a **UNet** architecture with an increased depth to capture higher-level semantic features.

- **DoubleConv Blocks:** Each stage utilizes dual $3 \times 3$ convolutions followed by **BatchNorm2d** and **ReLU** to stabilize training.
- **Skip Connections:** Encoder features are concatenated directly into the decoder to preserve spatial resolution for edge precision.
- **Bottleneck:** A 1024-channel latent space for complex feature extraction.

### 3. Video Inference Pipeline
Developed a custom **OpenCV-based engine** to process HD dashcam footage. The engine uses **GPU acceleration** (CUDA/MPS) to perform frame-by-frame segmentation and generates a side-by-side comparison video with a 30% alpha-blended overlay.

---

## 📂 Project Structure

```bash
autovision-cityscapes/
├── data/                  # Cityscapes raw image pairs
├── src/                   # Core source modules
│   ├── dataset.py         # Custom Cityscapes Dataset & Transforms
│   ├── model.py           # UNet architecture (DoubleConv, Up/Down blocks)
│   ├── utils.py           # Mask baking & visualization helpers
│   └── inference.py       # Video processing engine (FFmpeg integrated)
├── notebooks/
│   ├── 01_Training.ipynb  # Fine-tuning loop on GPU
│   └── 02_Evaluation.ipynb# mIoU calculation & "Showdown" visualizations
├── results/               # Demo videos and segmented screenshots
└── requirements.txt       # Project dependencies
