import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score)
import time

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Importazione Dati Preprocessati
X_train = np.load('X_train_pca.npy')
X_test = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# ============================================
# 2. BASELINE MODEL
# ============================================
# SVM con kernel Linear
print(f"\nTraining SVM baseline (kernel='linear', C=1.0) ...")
start_time = time.time()

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train, y_train)
train_time = time.time() - start_time
# Predizioni
y_train_pred = svm.predict(X_train)
y_test_pred = svm.predict(X_test)
# Valutazione
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"\n Baseline Model - Risultati:")
print(f"   Training time: {train_time:.2f} secondi")
print(f"   Training accuracy: {train_acc:.4f}")
print(f"   Test accuracy: {test_acc:.4f}")
print(f"   Overfitting gap: {(train_acc - test_acc):.4f}")
print("\n")
"""
 Da Preprocessing con LDA
 Baseline Model - Risultati:
   Training time: 23.58 secondi
   Training accuracy: 0.8385
   Test accuracy: 0.8379
   Overfitting gap: 0.0007
"""
# Valutazione: modello troppo semplice - UNDERFITTING
if train_acc < 0.9 and (train_acc - test_acc) < 0.01:
    print(f'\n Il modello lineare è troppo semplice. UNDERFITTING')


