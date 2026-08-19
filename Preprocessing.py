"""
Adult (Census Income) Dataset - Preprocessing
Fasi: Missing Values, Categorical data, Training-Test Set Split, Scaling, Dimensional Reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

# Si separa colonne numeriche e categoriche
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
categorical_cols = df.columns.difference(numeric_cols)

# Si imputa separatamente
imr_num = SimpleImputer(missing_values=np.nan, strategy="mean")
imr_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

df_clean = df.copy()
df_clean[numeric_cols] = imr_num.fit_transform(df[numeric_cols])
df_clean[categorical_cols] = imr_cat.fit_transform(df[categorical_cols])

# Esportazione Data Set senza Missing values
df_clean.to_pickle('data_clean.pkl')
print("\nDataset pulito salvato in 'data_clean.pkl'")


# ============================================
# 2. DEALING WITH CATEGORICAL DATA
# ============================================
# CLASS LABEL: INCOME
# trasformiamo i label di classe (<50k e >50k) in classi 0,1 e
# poi in dati numerici interi 0,1
class_mapping = {label: idx for idx, label in enumerate(np.unique(df_clean['income']))}
df_clean['income'] = df_clean['income'].map(class_mapping)

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

# Definisci le colonne categoriche da codificare
categorical_cols = ['workclass', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'native-country']
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
cols = df_encoded.columns.tolist()
print(f"\nColonne dopo encoding: {df_encoded.shape[1]}")
print(f"Cardinalità native-country: {df_clean['native-country'].nunique()}")

# Adesso si hanno 83 Features. 41 Native_Countries
# SPOSTIAMO LA COLONNA DELLA CLASSE 'INCOME' COME PRIMA COLONNA
df_encoded.insert(0, 'income', df_encoded.pop('income'))

# Esportazione Data Set Encoded
df_encoded.to_pickle('data_encoded.pkl')
print("\nDataset encoded salvato in 'data_encoded.pkl'")


# ============================================
# 3. TRAINING-TEST DATA SPLITTING
# ============================================
X, y = df_encoded.iloc[:, 1:].values, df_encoded.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)


# ============================================
# 4. SCALING
# ============================================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Salvataggio dei dati scalati, come file Numpy per LDA
np.save('X_train_std.npy', X_train_std)
np.save('X_test_std.npy', X_test_std)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

# Salva anche lo scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nDataset standardizzati salvati (.npy)")


# ============================================
# 5. FEATURE EXTRACTION: PCA - Visualizzazione
# ============================================
# Dati espressi nelle componenti principali (calcolati in altro file)
X_train_pca = np.load('X_train_pca.npy')
X_test_pca = np.load('X_test_pca.npy')

# ScatterPlot Dati proiettati sulle prime due componenti
colors = ['r', 'b']
markers = ['s', 'x']
sizes = [160, 80]
nomi_classi = ['<50k', '>50k']

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
plt.title('Proiezione PCA. Test_set data')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nGrafico PCA visualizzato")