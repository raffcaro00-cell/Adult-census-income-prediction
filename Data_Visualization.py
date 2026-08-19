"""
Adult (Census Income) Dataset - Esplorazione Iniziale
Dataset UCI: https://archive.ics.uci.edu/dataset/2/adult
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================
# 1. CARICAMENTO DATASET
# ============================================

# URL del dataset dall'UCI Repository
url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Nomi delle colonne
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Caricamento dati
df_train = pd.read_csv(
    url_train,
    names=column_names,
    sep=r',\s*',
    engine='python'
)

df_test = pd.read_csv(
    url_test,
    names=column_names,
    sep=r',\s*',
    engine='python'
)

# Combiniamo train e test per esplorazione completa
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"\n Dataset caricato:\n {df.shape[0]} righe, {df.shape[1]} colonne\n")


# ============================================
#  1. GUARDIAMO I DATI
# ============================================

print("=" * 60)
print("PRIME 5 RIGHE DEL DATASET")
print("=" * 60)
print(df.head())
print("\n")

print("=" * 60)
print("INFORMAZIONI GENERALI")
print("=" * 60)
print(df.info())
print("\n")

print("=" * 60)
print("STATISTICHE DESCRITTIVE PER FEATURE NUMERICHE")
print("=" * 60)
pd.set_option('display.max_columns', None)
print(df.describe(percentiles=[]))
print("\n")


# ============================================
#  2. ANALISI FEATURE
# ============================================

print("=" * 60)
print("TIPI DI FEATURE")
print("=" * 60)

# Feature numeriche
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
# Seleziona dati numerici. Seleziona il nome delle feature associate. Crea lista
print(f"\n Feature Numeriche ({len(numerical_features)}):")
print(numerical_features)

# Feature categoriche
categorical_features = df.select_dtypes(include=['object', 'string']).columns.tolist()
categorical_features.remove('income')  # Rimuoviamo il target
print(f"\n Feature Categoriche ({len(categorical_features)}):")
print(categorical_features)

print(f"\n Target variable: income")
print("\n")


# ============================================
# 3. MISSING VALUES
# ============================================

print("=" * 60)
print(" MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing Count': missing.values,
    'Percentage (%)': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
# Seleziona feature con dati mancanti. Ordine decrescente

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print(" Nessun missing value trovato!")
print("\n")

samples_with_missing = df.isnull().any(axis=1).sum()  # Conta righe con almeno un NaN
total_samples = len(df)
pct_samples_with_missing = (samples_with_missing / total_samples) * 100

print(f"\n Percentuale con campioni con almeno un dato mancante:\n {pct_samples_with_missing:.2f}%\n")


# ============================================
# 4. DISTRIBUZIONE TARGET
# ============================================

print("=" * 60)
print(" DISTRIBUZIONE CLASSE TARGET (income)")
print("=" * 60)

# Puliamo i valori del target (rimuovere punti finali e spazi se presenti)
df['income'] = df['income'].str.strip().str.rstrip('.')

# Contiamo il numero di valori per ogni classe e percentuali
target_dist = df['income'].value_counts()
target_pct = df['income'].value_counts(normalize=True) * 100

print("\nConteggio:")
print(target_dist)
print("\nPercentuale:")
print(target_pct)
print("\n")


# ============================================
# 5. CARDINALITÀ FEATURE CATEGORICHE
# ============================================

print("=" * 60)
print(" DISTRIBUZIONE FEATURE CATEGORICHE")
print("=" * 60)
print("(Numero di valori unici per ogni feature categorica)\n")

for col in categorical_features:
    n_unique = df[col].nunique()
    print(f"{col:20s}: {n_unique:3d} valori unici")
print("\n")


# Esportazione dati
df.to_pickle('data.pkl')