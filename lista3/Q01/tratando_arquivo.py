#Gerar a árvore para a base de dados Restaurante, alterando a codificação do atributo cliente para conter a
#seguinte codificação:
#Cliente_Nenhum = 0
#Cliente_Algum = 1
#Cliente_Cheio = 2

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle


base = pd.read_csv('lista3/Q01/restaurante.csv', sep=';')
base2 = pd.read_csv('lista3/Q01/restaurante.csv', sep=';', usecols=['Alternativo', 'Bar'])#filtrando as colunas Alternativo e Bar


print(base.head())#mostra as primeiras linhas do dataframe
#print('\n',base2.tail(2))#mostra as ultimas linhas do dataframe

Classificação = base.columns[-1]
np.unique(base[Classificação], return_counts=True)#mostra as classes e a quantidade de cada classe
print('\n',np.unique(base[Classificação], return_counts=True))

sns.countplot(x = base[Classificação]);#plota o grafico de barras da quantidade de cada classe
#plt.show()

#para codificar todos os atributos para laberEncoder de uma única vez
base_encoded = base.apply(LabelEncoder().fit_transform)#codifica todos os atributos
cols_label_encode = ['Alternativo', 'Bar', 'SexSab','fome', 'Cliente','Preco', 'Chuva', 'Res','Tempo']
base[cols_label_encode] = base[cols_label_encode].apply(LabelEncoder().fit_transform)
len(np.unique(base['Cliente']))#mostra a quantidade de classes do atributo Cliente
print('\n Quantidade de classes do atributo Cliente antes da alteração:\n',base.head())

# Alterando a codificação do Cliente 
# antes: Alguns=0, Cheio=1, Nenhum=2
# Nenhum(2)->0, Alguns(0)->1, Cheio(1)->2
base['Cliente'] = base['Cliente'].map({2: 0, 0: 1, 1: 2})
# depois: Cliente_Nenhum = 0, Cliente_Algum = 1, Cliente_Cheio = 2
print('\n Quantidade de classes do atributo Cliente após a alteração:\n',base.head())

cols_onehot_encode = ['Tipo']
# Inicializar o OneHotEncoder (sparse_output=False retorna um array denso)
onehot = OneHotEncoder(sparse_output=False)

# Aplicar o OneHotEncoder apenas nas colunas categóricas
df_onehot = onehot.fit_transform(base[cols_onehot_encode])

# Obter os novos nomes das colunas após a codificação
nomes_das_colunas = onehot.get_feature_names_out(cols_onehot_encode)

# Criar um DataFrame com os dados codificados e as novas colunas
df_onehot = pd.DataFrame(df_onehot, columns=nomes_das_colunas)

# Combinar as colunas codificadas com as colunas que não foram transformadas
base_encoded= pd.concat([df_onehot, base.drop(columns=cols_onehot_encode)], axis=1)

#print('\n',base_encoded.head())
#print('\n',base_encoded.shape)#mostra a quantidade de linhas e colunas do dataframe

# Supondo que a última coluna seja o target
X_prev= base_encoded.iloc[:, :-1]#pega todas as colunas menos a última
y_classe = base_encoded.iloc[:, -1]#pega a última coluna

#print('\n',X_prev.head())
#print('\n',y_classe.head())
#print('\n',y_classe.shape)#mostra a quantidade de linhas e colunas do dataframe

X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 42)#divide o dataframe em treinamento e teste
X_treino.shape
X_teste.shape
#print('\n',X_teste.head())

#print('\n',y_treino.head())
#print('\n',y_teste.shape)

with open('Restaurante.pkl', mode = 'wb') as f:#salva o dataframe em um arquivo
  pickle.dump([X_treino, X_teste, y_treino, y_teste], f)