# MLP Classifiers for Visual Pattern Recognition

This repository contains three machine learning projects using multi-layer perceptrons (MLPs) built with `scikit-learn`. Each project focuses on a different classification task—ranging from synthetic data to real-world handwritten characters. These were completed as part of the final coursework for ITP 449: Machine Learning at USC.

## Project Descriptions

### Project 1: Spiral Data Classification
This project generates a synthetic dataset of two interleaved spiral-shaped classes to demonstrate binary classification using an MLP. The goal is to train a model that can accurately distinguish between the two classes despite their nonlinear structure.

**Key steps:**
- Creates a labeled dataset by computing spiral coordinates with noise.
- Visualizes the data distribution using a scatter plot.
- Splits the data into training and test sets with class stratification.
- Trains a binary `MLPClassifier` with two hidden layers using scikit-learn.
- Evaluates the model using test accuracy and a confusion matrix.
- Visualizes the loss curve during training and plots the learned decision boundary.

**File:** `FinalProject_1_WangAmy.py`

---

### Project 2: Handwritten Letter Classification (A–Z)
This project trains an MLP classifier to recognize uppercase English letters from the `A_Z_Handwritten_Data.csv` dataset, which contains grayscale pixel data for letters labeled from 0 to 25. The labels are remapped to characters A–Z to improve interpretability.

**Key steps:**
- Loads the dataset and maps numerical labels to their corresponding letters.
- Splits the data into training and testing sets with stratification.
- Scales the pixel values to the 0–1 range for better training performance.
- Trains a multi-class `MLPClassifier` with three hidden layers using scikit-learn.
- Evaluates model performance using accuracy metrics and a confusion matrix.
- Displays a correctly predicted test image and highlights one misclassified example for analysis.

**File:** `FinalProject_2_WangAmy.py`

---

### Project 3: Alphanumeric Classification and Phrase Recognition

This project combines two datasets—handwritten letters (A–Z) and digits (0–9)—to train a single MLP model capable of classifying 36 distinct characters. After training, the model is used to interpret a phrase from a grayscale image by slicing the image into individual characters and predicting them one at a time.

**Key steps:**
- Maps labels in both datasets (letters and digits) to string characters for unified classification.
- Merges training and test sets from the A–Z and MNIST datasets.
- Visualizes class distribution and trains a multi-class `MLPClassifier` using scikit-learn.
- Evaluates model performance with accuracy metrics and a confusion matrix.
- Applies the model to read a 6-character handwritten phrase from an input image (`testPhrase.png`) using manual slicing and grayscale conversion.

**File:** `FinalProject_3_WangAmy.py`

---

## Dependencies

- Python 3.8+
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn

You can install dependencies with:

```bash
pip install -r requirements.txt

