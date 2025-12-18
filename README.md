# Real-Time-ASL-Alphabet-Recognition
# Real-Time American Sign Language Alphabet Recognition

This repository contains the code and experiments for the EE526 Final Project on
American Sign Language (ASL) alphabet recognition using both traditional machine
learning methods and deep learning.

The project studies how different classification approaches generalize to unseen
ASL data and demonstrates an end-to-end inference pipeline for both image and
video inputs.

---

## Project Overview

- Task: ASL alphabet classification (static gestures)
- Models:
  - K-Nearest Neighbors (MediaPipe landmarks)
  - Logistic Regression (MediaPipe landmarks)
  - Decision Tree (MediaPipe landmarks)
  - YOLOv8 classification model (RGB images)
- Evaluation:
  - Training on a public ASL dataset
  - Validation on a separate unseen dataset
  - Inference on saved video and live camera input

---

## Repository Structure

- `scripts/`  
  Python scripts for data preparation, training, evaluation, and inference

- `asl_dataset/`  
  Description of the public ASL training dataset (not included in repository)

- `data/`  
  Description of the locally collected unseen ASL dataset (not included)


- `outputs/`  
  Plots, confusion matrices, and inference videos

- `report/`  
  Final project report in IEEE format

---

## Setup

### 1. Create virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
