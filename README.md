# Adult Census Income Prediction: End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20SVM-orange)

## 📌 Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to classify demographic data from the UCI Adult Census Income dataset. The goal is to predict whether an individual's annual income exceeds $50,000 based on census parameters.

Developed as part of a research-oriented computational portfolio, this project emphasizes **clean code structure, modular preprocessing, hyperparameter optimization, and model evaluation**.

---

## 🛠️ Pipeline Architecture & Methodology

The project is structured into modular Python components:
- **Data Preprocessing & Encoding (`Preprocessing.py`, `categorical_data.py`, `missing_data.py`):** Handling missing values, categorical encoding, and feature scaling (`splitting_scaling.py`).
- **Dimensionality Reduction (`PCA_feature_extraction.py`):** Principal Component Analysis (PCA) for spatial feature transformation and dimensionality reduction.
- **Model Training & Optimization (`svm_baseline.py`, `hyperparameters_optimization.py`):** Support Vector Machine (SVM) baselines and grid search hyperparameter tuning.
- **Validation & Diagnostics (`cross_validation.py`, `learning_curves.py`, `validation_curve.py`):** K-Fold Cross-Validation, learning curves, and validation curves to assess overfitting/underfitting.
- **Pipeline Execution (`main.py`):** Central entry point executing the workflow end-to-end.

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/raffcaro00-cell/Adult-census-income-prediction.git](https://github.com/raffcaro00-cell/Adult-census-income-prediction.git)
   cd Adult-census-income-prediction
