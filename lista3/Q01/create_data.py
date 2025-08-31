import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Carregar os dados do CSV
base = pd.read_csv('restaurante.csv', sep=';')

# Codificar os atributos categóricos
cols_label_encode = ['Alternativo', 'Bar', 'SexSab','fome', 'Cliente','Preco', 'Chuva', 'Res','Tempo', 'Tipo']
base[cols_label_encode] = base[cols_label_encode].apply(LabelEncoder().fit_transform)

# Separar features (X) e target (y)
X = base.drop('Conclusao', axis=1)  # Todas as colunas exceto 'Conclusao'
y = base['Conclusao']  # Apenas a coluna 'Conclusao'

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Salvar os dados no arquivo pickle
with open('Restaurante.pkl', 'wb') as f:
    pickle.dump((X_treino, X_teste, y_treino, y_teste), f)

print("Dados salvos em Restaurante.pkl")
print(f"Forma dos dados de treino: {X_treino.shape}")
print(f"Forma dos dados de teste: {X_teste.shape}")
print(f"Colunas: {list(X_treino.columns)}")
print(f"Classes: {y_treino.unique()}")
