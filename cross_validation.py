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
# 3. K-FOLD CROSS VALIDATION
# ============================================

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import numpy as np

import time

start_time = time.time()
# SVM con kernel RBF
svm = SVC(kernel='rbf', C=1.0, random_state=0)

# StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

scores = []
for k, (train, evaluation) in enumerate(kfold.split(X_train, y_train)):
    svm.fit(X_train[train], y_train[train])
    score = svm.score(X_train[evaluation], y_train[evaluation])
    scores.append(score)
    print(f'Fold: {k + 1}, Class dist.: {np.bincount(y_train[train])}, Acc: {score:.3f}')
train_time = time.time() - start_time

print(f'CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
print(f'Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}')
print(f'tempo di addestramento: {train_time}')
"""
Da LDA
Fold: 1, Class dist.: [26751  8414], Acc: 0.841
Fold: 2, Class dist.: [26751  8414], Acc: 0.844
Fold: 3, Class dist.: [26751  8415], Acc: 0.836
Fold: 4, Class dist.: [26751  8415], Acc: 0.844
Fold: 5, Class dist.: [26752  8414], Acc: 0.834
CV accuracy: 0.840 +/- 0.004
Min: 0.834, Max: 0.844

"""

