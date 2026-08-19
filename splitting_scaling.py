"""
Adult (Census Income) Dataset - Train/Test Split e Scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Importazione Data Set
df_split = pd.read_pickle('data_encoded.pkl')

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


# ============================================
# 3. TRAINING-TEST DATA SPLITTING
# ============================================
print("Splitting dataset in Training e Test set")
print("-" * 70)

X, y = df_split.iloc[:, 1:].values, df_split.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=0,
    stratify=y
)

print(f"\nDimensioni dataset:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_test:  {y_test.shape}")
print("\n")


# ============================================
# 4. SCALING
# ============================================
print("Standardizzazione features")
print("-" * 70)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

print(f"\nDimensioni dataset standardizzati:")
print(f"  X_train_std: {X_train_std.shape}")
print(f"  X_test_std:  {X_test_std.shape}")
print("\n")


# ============================================
#  SALVATAGGIO DATI
# ============================================
# Salvataggio dei dati come file Numpy
np.save('X_train_std.npy', X_train_std)
np.save('X_test_std.npy', X_test_std)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Salva anche lo scaler per uso futuro
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("File salvati:")
print("  - X_train_std.npy")
print("  - X_test_std.npy")
print("  - y_train.npy")
print("  - y_test.npy")
print("  - scaler.pkl")
