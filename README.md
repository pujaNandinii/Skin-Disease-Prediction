# Skin Disease Detection using Deep Learning

This project detects different types of skin diseases from images using a Deep Learning model based on EfficientNetB0.

A Tkinter GUI application is provided where the user can upload an image and get the predicted skin disease.

## Features

- Transfer Learning using EfficientNetB0
- Proper image preprocessing
- Handling class imbalance using class weights
- GUI application for easy prediction
- Model saved in .keras format

## Dataset

Dataset used: Skin Cancer: Malignant vs Benign (Kaggle)
It has,
1. benign (healthy)
2. malignant (diseased)
Dataset link: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign/

## How to Run

1. Install required packages

pip install -r requirements.txt

2. Run the application

python app.py

3. Upload an image and click "Detect Disease"

## Requirements

See requirements.txt


