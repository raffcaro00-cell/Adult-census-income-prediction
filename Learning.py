"""
Adult (Census Income) Dataset - Learning
Dataset UCI: https://archive.ics.uci.edu/dataset/2/adult
Algoritmo: Support Vector Machine (SVM)
Fasi: Model Selection, Hyperparameter Optimization, Cross Validation, Confronto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, learning_curve

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================
# 1. CARICAMENTO DATI
# ============================================
print(" STEP 1: Caricamento Dati Preprocessati")
print("-" * 80)

# Importazione Dati Preprocessati
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"   Dati caricati:")
print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"   Distribuzione classi training: {np.bincount(y_train)}")
print(f"   Distribuzione classi test: {np.bincount(y_test)}")
print("\n")

# ============================================
# 2. BASELINE MODEL
# ============================================

# SVM CON KERNEL = LINEAR
print(f"\nTraining SVM baseline (kernel='linear', C=1.0) ...")

print(f' Baseline Model - Risultati: \nTraining accuracy: 0.8385 - \nTest accuracy: 0.8379 - \nOverfitting gap: 0.0007')
print(f'\n Il modello lineare è troppo semplice. UNDERFITTING')

# ============================================
# 3. K-FOLD CROSS VALIDATION
# ============================================

# SVM CON KERNEL = RBF


# ============================================
# 4. Validation Curve Per Gamma da 10^(-5) a 10^(5)
# ============================================

