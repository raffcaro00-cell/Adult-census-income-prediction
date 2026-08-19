"""
Adult (Census Income) Dataset - Encoding Categorical Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importazione Data Set senza Missing values
df_clean = pd.read_pickle('data_clean.pkl')

# Configurazione visualizzazioni
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


# ============================================
# 2.1. DEALING WITH CATEGORICAL DATA
# ============================================

# CLASS LABEL: INCOME
# Trasformiamo i label di classe (<50k e >50k) in classi 0,1
class_mapping = {label: idx for idx, label in enumerate(np.unique(df_clean['income']))}
df_clean['income'] = df_clean['income'].map(class_mapping)

# Verifica
print("Valori unici target 'income':")
print(df_clean['income'].unique())
print("\n")


# ORDINALE: EDUCATION
education_mapping = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
    '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9,
    'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12,
    'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16
}
df_clean['education'] = df_clean['education'].map(education_mapping)

# OSSERVAZIONE: diventa indistinguibile dalla feature education_num
# Per cui eliminiamo la colonna
df_clean = df_clean.drop(columns=['education'])


# NOMINALE: WORK-CLASS, MARITAL-STATUS, OCCUPATION, RELATIONSHIP, RACE, SEX, NATIVE-COUNTRY
# One-Hot Encoding per le feature categoriche nominali
categorical_cols = ['workclass', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country']

df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
cols = df_encoded.columns.tolist()

"""
# Visualizziamo le colonne (opzionale)
n = 5  # elementi per riga
for i in range(0, len(cols), n):
    print(cols[i:i+n])
print(f"\nTotale colonne: {len(cols)}")
"""


# ============================================
# 2.2. ANALISI DISTRIBUZIONE NATIVE-COUNTRY
# ============================================
# Contiamo campioni per Paese
# Creiamo lista per country
onehot_cols = [c for c in df_encoded.columns if c.startswith("native-country_")]
# E li sommiamo
feature_counts = df_encoded[onehot_cols].sum().sort_values(ascending=False)

print("Distribuzione campioni per native-country:")
print(feature_counts)
print("\n")

# Valutazione: meglio non accorpare insieme gli Stati


# ============================================
# 2.3. RIORDINO COLONNE
# ============================================
# Vediamo dove sta la colonna di classe
print(f"Posizione iniziale colonna 'income': {df_encoded.columns.get_loc('income')}")

# La spostiamo in prima posizione
df_encoded.insert(0, 'income', df_encoded.pop('income'))

print(f"Posizione finale colonna 'income': {df_encoded.columns.get_loc('income')}")
print("\n")


# Esportazione Data Set Encoded
df_encoded.to_pickle('data_encoded.pkl')
print("Dataset encoded salvato in 'data_encoded.pkl'")
print(f"Dimensioni finali: {df_encoded.shape[0]} righe, {df_encoded.shape[1]} colonne")