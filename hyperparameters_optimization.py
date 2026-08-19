import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import numpy as np

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

# Importazione Dati Preprocessati
X_train = np.load('X_train_lda.npy')
X_test = np.load('X_test_lda.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# ============================================
# 5. HYPERPARAMETERS_OPTIMIZATION
# ============================================
# Tuning hyperparameters via grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

# Carica i dati PCA
X_train_pca = np.load('X_train_pca.npy')
X_test_pca = np.load('X_test_pca.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
# Range molto ampio per i parametri
param_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
# Griglia di parametri
param_grid = [
    {'C': param_range,
     'kernel': ['linear']},
    {'C': param_range,
     'gamma': param_range,
     'kernel': ['rbf']},
    {'C': param_range,
     'gamma': param_range,
     'kernel': ['sigmoid']}
]
# GridSearchCV
print("Inizio Grid Search")
gs = GridSearchCV(estimator=SVC(random_state=0),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1,
                  verbose=2)
gs = gs.fit(X_train_pca, y_train)
print(f"\nMiglior score (validation): {gs.best_score_:.6f}")
print(f"Migliori parametri: {gs.best_params_}")
# Valutazione sul test set
test_score = gs.score(X_test_pca, y_test)
print(f"Score sul test set: {test_score:.6f}")
# Dettagli del miglior modello
print(f"\nMiglior estimator: {gs.best_estimator_}")

