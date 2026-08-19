"""
Adult (Census Income) Dataset - Gestione Missing Values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# Importazione Data Set
df = pd.read_pickle('data.pkl')

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


# ============================================
# 1. GESTIONE MISSING VALUES
# ============================================
print("Gestione Missing Values. 7.41% dei campioni")
print("-" * 70)

# Separa colonne numeriche e categoriche
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']

# Seleziona solo le colonne categoriche (tipo object/string)
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

# Imputa separatamente
imr_num = SimpleImputer(missing_values=np.nan, strategy="mean")
imr_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

df_clean = df.copy()
df_clean[numeric_cols] = imr_num.fit_transform(df[numeric_cols])
df_clean[categorical_cols] = imr_cat.fit_transform(df[categorical_cols])

# Esportazione Data Set senza Missing values
df_clean.to_pickle('data_clean.pkl')
print("\nDataset pulito salvato in 'data_clean.pkl'")


# ============================================
# 2. VERIFICA MISSING VALUES
# ============================================
missing = df_clean.isnull().sum()
missing_pct = (missing / len(df_clean)) * 100
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing Count': missing.values,
    'Percentage (%)': missing_pct.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

# Seleziona feature con dati mancanti. Ordine decrescente
if len(missing_df) > 0:
    print("\nMissing values rimanenti:")
    print(missing_df.to_string(index=False))
else:
    print("\nNessun missing value trovato!")

print("\n")


# ============================================
# 3. VERIFICA TIPI DI FEATURE
# ============================================
# Feature numeriche
numerical_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
print(f"Feature Numeriche ({len(numerical_features)}):")
print(numerical_features)

# Feature categoriche
categorical_features = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
categorical_features.remove('income')  # Rimuoviamo il target
print(f"\nFeature Categoriche ({len(categorical_features)}):")
print(categorical_features)
print("\n")