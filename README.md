# 🎗️ Histopathologic Cancer Detection Project 🎗️

Welcome to the **Histopathologic Cancer Detection Project**! This project aims to classify histopathology images as either cancerous or non-cancerous using a Convolutional Neural Network (CNN) built in PyTorch. This README will walk you through the structure, methods, and key concepts used in building and training the model.

## 🌟 Objective

The objective of this project is to build a high-performance classifier for cancer detection from histopathologic images.

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Overview](#data-overview)
- [Model Architecture](#model-architecture)
- [Loss & Optimization](#loss--optimization)
- [GPU Support](#gpu-support)
- [Evaluation](#evaluation)
- [Challenges & Learnings](#challenges--learnings)
- [Acknowledgments](#acknowledgments)

---


---

## 🚀 Getting Started

To set up and run this project on your machine, ensure you have Python and PyTorch installed. Clone the repository and navigate to the project directory.

Clone the repository
git clone https://github.com/YourUsername/Histopathologic_Cancer_Detection.git cd Histopathologic_Cancer_Detection

Install required libraries
pip install -r requirements.txt


---

## 📊 Data Overview

### Dataset Details
The dataset consists of histopathology images labeled as either containing cancerous tissue or not:

- `0`: No Cancer
- `1`: Cancer

---

## 🔥 Model Architecture

Our CNN architecture uses 5 convolutional layers with batch normalization and ReLU activation functions. The layers increase in depth, starting from 32 channels up to 512 channels, capturing increasingly complex patterns.

### Layer Breakdown
- **Convolutional Layers**: Feature extraction from images.
- **MaxPooling Layers**: Reducing spatial dimensions, focusing on important features.
- **Dropout Layers**: Preventing overfitting by randomly dropping nodes.
- **Fully Connected Layers**: Classification based on extracted features.


### Why These Layers? 🤖
- **ReLU Activation**: Prevents vanishing gradient issues and accelerates learning.
- **Batch Normalization**: Normalizes layer outputs for faster convergence and better performance.
- **Dropout**: Regularization technique to improve model generalization.

---

## 🧮 Loss & Optimization

We use Binary Cross-Entropy Loss (BCELoss) for this binary classification task, with the Adam Optimizer to minimize the loss effectively.


---

## 🌐 GPU Support

Training on GPU (if available) is enabled for faster computation. Ensure CUDA is enabled in your PyTorch installation.


---

## 📈 Evaluation

Each epoch, we track:

- **Training Loss**: Measures error in the training set.
- **Validation Loss**: Measures error in the validation set.
- **Validation AUC**: Provides a measure of how well the model distinguishes between classes.

We save the model if the validation loss decreases:


---

## 🤔 Challenges & Learnings

- **Class Imbalance**: Balanced sampling prevents model bias.
- **Computational Power**: Utilizing GPU enables faster training.

---

## 📑 Acknowledgments

This project was inspired by the Kaggle Histopathologic Cancer Detection competition, and we would like to thank the contributors for providing this valuable dataset.


