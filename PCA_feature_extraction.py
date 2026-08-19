"""
Adult (Census Income) Dataset - Feature Extraction con PCA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


# ============================================
# 5.1. CARICAMENTO DATI
# ============================================
print("Caricamento dati standardizzati")
print("-" * 70)

# Carica i dati già standardizzati
X_train_std = np.load('X_train_std.npy')
X_test_std = np.load('X_test_std.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Verifica le dimensioni
print(f"\nDimensioni dataset:")
print(f"  X_train: {X_train_std.shape}")
print(f"  X_test:  {X_test_std.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test:  {y_test.shape}")

d = X_train_std.shape[1]  # numero di feature
print(f"\nNumero di feature: {d}")
print("\n")


# ============================================
# 5.2. CALCOLO MATRICE DI COVARIANZA
# ============================================
print("Calcolo matrice di covarianza")
print("-" * 70)

cov_matrix = np.cov(X_train_std.T)
print(f"Dimensioni matrice di covarianza: {cov_matrix.shape[0]}x{cov_matrix.shape[1]}")
print("\n")


# ============================================
# 5.3. CALCOLO AUTOVALORI E AUTOVETTORI
# ============================================
print("Calcolo autovalori e autovettori")
print("-" * 70)

# Calcolo degli autovalori e autovettori dalla matrice di covarianza
eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)

# Creazione di coppie (valore assoluto autovalore, autovettore)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Ordinamento delle coppie in ordine decrescente rispetto agli autovalori
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Stampa degli autovalori in ordine decrescente
print('\nAutovalori in ordine decrescente:')
for eigen_val in eigen_pairs:
    print(f"  {eigen_val[0]:.6f}")
print("\n")


# ============================================
# 5.4. ANALISI DISCRIMINABILITY
# ============================================
print("Analisi discriminability")
print("-" * 70)

# Calcolo della varianza cumulativa
total_variance = sum([autoval[0] for autoval in eigen_pairs])
discr = [(autoval[0] / total_variance) for autoval in eigen_pairs]
cumulative_variance = np.cumsum(discr)

# Visualizzazione della discriminabilità
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(discr) + 1), discr, alpha=0.5, align='center',
        label='Individual "Discriminability"')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
         label='Cumulative Variance')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Componenti Principali')
plt.legend(loc='best')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Selezione del numero di componenti che spiegano il 95% della varianza
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNumero di componenti per spiegare il 95% della varianza: {n_components}")
print(f"Discriminabilità con {n_components} componenti: {cumulative_variance[n_components - 1]:.4f}")
print("\n")


# ============================================
# 5.5. COSTRUZIONE MATRICE DI TRASFORMAZIONE
# ============================================
print("Costruzione matrice di trasformazione W")
print("-" * 70)

# Definiamo W, matrice del cambio base. Ora è d x n_components
w_matrix = np.column_stack([eigen_pairs[i][1].real for i in range(n_components)])
print(f"Dimensioni matrice W: {w_matrix.shape}")
print("\n")


# ============================================
# 5.6. PROIEZIONE DATASET
# ============================================
print("Proiezione dataset sulle componenti principali")
print("-" * 70)

# Esprimiamo il Dataset X_train_std nella base degli autovettori
X_train_pca = X_train_std.dot(w_matrix)
X_test_pca = X_test_std.dot(w_matrix)

print(f"Dimensioni X_train_pca: {X_train_pca.shape}")
print(f"Dimensioni X_test_pca: {X_test_pca.shape}")
print("\n")


# ============================================
# 5.7. VISUALIZZAZIONE PCA - TRAINING SET
# ============================================
colors = ['r', 'b']
markers = ['s', 'x']
sizes = [160, 80]
nomi_classi = ['<50k', '>50k']

plt.figure(figsize=(10, 6))
for l, c, s, m, income in zip(np.unique(y_train), colors, sizes, markers, nomi_classi):
    plt.scatter(
        X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c,
        s=s,
        label=income,
        marker=m,
    )
plt.xlabel('Prima Componente Principale')
plt.ylabel('Seconda Componente Principale')
plt.title('Proiezione PCA - Training Set')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 5.8. VISUALIZZAZIONE PCA - TEST SET
# ============================================
plt.figure(figsize=(10, 6))
for l, c, s, m, income in zip(np.unique(y_train), colors, sizes, markers, nomi_classi):
    plt.scatter(
        X_test_pca[y_test == l, 0],
        X_test_pca[y_test == l, 1],
        c=c,
        s=s,
        label=income,
        marker=m,
    )
plt.xlabel('Prima Componente Principale')
plt.ylabel('Seconda Componente Principale')
plt.title('Proiezione PCA - Test Set')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 5.9. SALVATAGGIO DATI TRASFORMATI
# ============================================
np.save('X_train_pca.npy', X_train_pca)
np.save('X_test_pca.npy', X_test_pca)

print("File salvati:")
print("  - X_train_pca.npy")
print("  - X_test_pca.npy")