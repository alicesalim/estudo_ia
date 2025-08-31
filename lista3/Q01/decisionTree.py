import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree

# Carregar os dados salvos
with open('/Users/alice/estudos/estudo_ia/Restaurante.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Criar e treinar o modelo
modelo = DecisionTreeClassifier(criterion='entropy')
modelo.fit(X_treino, y_treino)

# Fazer previsões
previsoes = modelo.predict(X_teste)

# Mostrar resultados
print("\nPrevisões:", previsoes)
print("\nValores reais:", y_teste.head().to_list())


print('\n',accuracy_score(y_teste,previsoes))
print('\n',confusion_matrix(y_teste,previsoes))

cm = confusion_matrix(y_teste, previsoes)
print('\n',cm)
print('\n',classification_report(y_teste,previsoes))


previsores = X_treino.columns
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(13,13))
tree.plot_tree(modelo, feature_names=previsores, class_names = modelo.classes_, filled=True);
plt.show()
#mostra a árvore de decisão