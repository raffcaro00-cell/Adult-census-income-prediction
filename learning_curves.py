import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


#######





# Crea la pipeline con SVM usando i migliori parametri trovati da GridSearchCV
pipe_svm = Pipeline([
    ('scl', StandardScaler()),
    ('clf', SVC(C=gs.best_params_['C'],
                gamma=gs.best_params_['gamma'],
                kernel='rbf',  # o il kernel che hai usato
                random_state=0))
])

# Calcola le learning curves
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_svm,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# Calcola medie e deviazioni standard
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5,
         label='Training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])  # Adatta questo range se necessario
plt.title(f'Learning Curves - SVM (C={gs.best_params_["C"]}, gamma={gs.best_params_["gamma"]})')
plt.tight_layout()
plt.show()