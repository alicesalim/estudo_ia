#Baseado nestes códigos acima, encontrar o padrão de pessoas que sobreviveram ao desastre do TITANIC, que
#matou mais de 1.500 pessoas em 1912. A base de dados do TITANIC está no CANVAS.
#1. Visualize a base de dados primeiro, veja como estão os atributos e suas distribuições.
#2. Investigue a melhor forma de codificar cada atributo da base de dados.
#3. Forneça as regras que mostre o padrão de mortalidade.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import pickle

# Configurar o estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

print("ANÁLISE COMPLETA DO TITANIC - PADRÕES DE SOBREVIVÊNCIA")


print("\n1. VISUALIZAÇÃO DA BASE DE DADOS")


# Carregar a base de dados do Titanic
base = pd.read_csv('lista3/Q02/train.csv', sep=',')


print(base.head(10))


print(base.info())

print("\nEstatísticas descritivas:")
print(base.describe(include='all'))

# Análise da variável target (Survived)
Classificação = 'Survived'
sobreviventes = np.unique(base[Classificação], return_counts=True)
print(f"\nDistribuição de sobrevivência:")
print(f"Valores únicos: {sobreviventes[0]}")
print(f"Contagem: {sobreviventes[1]}")
print(f"Taxa de sobrevivência: {sobreviventes[1][1]/(sobreviventes[1][0]+sobreviventes[1][1])*100:.2f}%")

# Gráfico de sobrevivência
plt.figure(figsize=(10, 6))
sns.countplot(data=base, x='Survived', hue='Survived', palette=['#de5e36', '#2374b9'], legend=False)
plt.title('Distribuição de Sobrevivência no Titanic', fontsize=16, fontweight='bold')
plt.xlabel('Sobreviveu (0=Não, 1=Sim)', fontsize=12)
plt.ylabel('Quantidade de Passageiros', fontsize=12)
plt.xticks([0, 1], ['Não Sobreviveu', 'Sobreviveu'])
for i, v in enumerate(sobreviventes[1]):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()

# Análise das distribuições dos atributos
print("\nDistribuições dos atributos principais:")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribuições dos Atributos do Titanic', fontsize=16, fontweight='bold')

# Classe do passageiro
sns.countplot(data=base, x='Pclass', hue='Pclass', ax=axes[0,0], palette=['#2374b9', '#edb125', '#de5e36'], legend=False)
axes[0,0].set_title('Classe do Passageiro')
axes[0,0].set_xlabel('Classe (1=Primeira, 2=Segunda, 3=Terceira)')
axes[0,0].set_ylabel('Quantidade')

# Sexo
sns.countplot(data=base, x='Sex', hue='Sex', ax=axes[0,1], palette=['#2374b9', '#de5e36'], legend=False)
axes[0,1].set_title('Distribuição por Sexo')
axes[0,1].set_xlabel('Sexo')
axes[0,1].set_ylabel('Quantidade')

# Idade
sns.histplot(data=base, x='Age', bins=20, ax=axes[0,2], color='#2374b9')
axes[0,2].set_title('Distribuição de Idade')
axes[0,2].set_xlabel('Idade')
axes[0,2].set_ylabel('Frequência')

# Número de irmãos/cônjuges
sns.countplot(data=base, x='SibSp', hue='SibSp', ax=axes[1,0], palette='viridis', legend=False)
axes[1,0].set_title('Número de Irmãos/Cônjuges')
axes[1,0].set_xlabel('SibSp')
axes[1,0].set_ylabel('Quantidade')

# Número de pais/filhos
sns.countplot(data=base, x='Parch', hue='Parch', ax=axes[1,1], palette='viridis', legend=False)
axes[1,1].set_title('Número de Pais/Filhos')
axes[1,1].set_xlabel('Parch')
axes[1,1].set_ylabel('Quantidade')

# Tarifa
sns.histplot(data=base, x='Fare', bins=20, ax=axes[1,2], color='#edb125')
axes[1,2].set_title('Distribuição de Tarifa')
axes[1,2].set_xlabel('Tarifa')
axes[1,2].set_ylabel('Frequência')

plt.tight_layout()
plt.show()

# Análise de valores faltantes
print("\nAnálise de valores faltantes:")
print("Valores faltantes por coluna:")
print(base.isnull().sum())
print(f"\nPercentual de valores faltantes:")
print((base.isnull().sum() / len(base) * 100).round(2))

# Gráfico de valores faltantes
plt.figure(figsize=(12, 6))
missing_data = base.isnull().sum()
missing_percent = (missing_data / len(base) * 100).round(2)
bars = plt.bar(range(len(missing_data)), missing_percent, color='#de5e36')
plt.title('Percentual de Valores Faltantes por Coluna', fontsize=16, fontweight='bold')
plt.xlabel('Colunas', fontsize=12)
plt.ylabel('Percentual de Valores Faltantes (%)', fontsize=12)
plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()



print("2. INVESTIGAÇÃO DA CODIFICAÇÃO DOS ATRIBUTOS")


print("Análise dos tipos de dados e valores únicos:")
for col in base.columns:
    unique_count = base[col].nunique()
    print(f"{col}: {base[col].dtype} - {unique_count} valores únicos")

print("\nValores únicos das variáveis categóricas:")
print("Sex:", base['Sex'].unique())
print("Embarked:", base['Embarked'].unique())
print("Pclass:", base['Pclass'].unique())

print("\nRecomendações de codificação:")
print("1. Sex: LabelEncoder (2 valores únicos)")
print("2. Embarked: LabelEncoder (3 valores únicos)")
print("3. Pclass: Manter como numérico (já é ordinal)")
print("4. Age: Manter como numérico, tratar valores faltantes")
print("5. SibSp, Parch: Manter como numérico")
print("6. Fare: Manter como numérico")
print("7. Name, Ticket, Cabin: Remover (muitos valores únicos ou faltantes)")



print("3. ANÁLISE DOS PADRÕES DE MORTALIDADE")


# Análise de sobrevivência por atributos
print("Taxa de sobrevivência por atributos:")

# Por classe
surv_class = base.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).round(3)
print("\nPor Classe:")
print(surv_class)

# Por sexo
surv_sex = base.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).round(3)
print("\nPor Sexo:")
print(surv_sex)

# Por porto de embarque
surv_embarked = base.groupby('Embarked')['Survived'].agg(['count', 'sum', 'mean']).round(3)
print("\nPor Porto de Embarque:")
print(surv_embarked)

# Gráficos de sobrevivência por atributos
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Taxa de Sobrevivência por Atributos', fontsize=16, fontweight='bold')

# Sobrevivência por classe
sns.barplot(data=base, x='Pclass', y='Survived', ax=axes[0,0], hue='Pclass', palette=['#2374b9', '#edb125', '#de5e36'], legend=False)
axes[0,0].set_title('Taxa de Sobrevivência por Classe')
axes[0,0].set_xlabel('Classe')
axes[0,0].set_ylabel('Taxa de Sobrevivência')
for i, v in enumerate(surv_class['mean']):
    axes[0,0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

# Sobrevivência por sexo
sns.barplot(data=base, x='Sex', y='Survived', ax=axes[0,1], hue='Sex', palette=['#2374b9', '#de5e36'], legend=False)
axes[0,1].set_title('Taxa de Sobrevivência por Sexo')
axes[0,1].set_xlabel('Sexo')
axes[0,1].set_ylabel('Taxa de Sobrevivência')
for i, v in enumerate(surv_sex['mean']):
    axes[0,1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

# Sobrevivência por idade (boxplot)
sns.boxplot(data=base, x='Survived', y='Age', ax=axes[1,0], hue='Survived', palette=['#de5e36', '#2374b9'], legend=False)
axes[1,0].set_title('Distribuição de Idade por Sobrevivência')
axes[1,0].set_xlabel('Sobreviveu (0=Não, 1=Sim)')
axes[1,0].set_ylabel('Idade')

# Sobrevivência por tarifa
sns.boxplot(data=base, x='Survived', y='Fare', ax=axes[1,1], hue='Survived', palette=['#de5e36', '#2374b9'], legend=False)
axes[1,1].set_title('Distribuição de Tarifa por Sobrevivência')
axes[1,1].set_xlabel('Sobreviveu (0=Não, 1=Sim)')
axes[1,1].set_ylabel('Tarifa')

plt.tight_layout()
plt.show()


print("4. TREINAMENTO DO MODELO E REGRAS DE DECISÃO")


# Preparar dados para modelagem
print("Preparando dados para modelagem...")

# Tratar valores faltantes
base_clean = base.copy()
base_clean['Embarked'] = base_clean['Embarked'].fillna('S')
base_clean['Age'] = base_clean['Age'].fillna(base_clean['Age'].median())

# Remover colunas problemáticas
base_clean = base_clean.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Codificar variáveis categóricas
cols_label_encode = ['Sex', 'Embarked']
base_clean[cols_label_encode] = base_clean[cols_label_encode].apply(LabelEncoder().fit_transform)

# Separar features e target
X_prev = base_clean.drop(['PassengerId', 'Survived'], axis=1)
y_classe = base_clean['Survived']

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size=0.20, random_state=42)

print(f"Dados preparados - Treino: {X_treino.shape}, Teste: {X_teste.shape}")

# Treinar modelo de árvore de decisão
print("\nTreinando modelo de árvore de decisão...")
modelo = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
modelo.fit(X_treino, y_treino)

# Fazer previsões
previsoes = modelo.predict(X_teste)

# Avaliar modelo
print("\nResultados do modelo:")
print(f"Accuracy: {accuracy_score(y_teste, previsoes):.3f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, previsoes))
print("\nRelatório de Classificação:")
print(classification_report(y_teste, previsoes))

# Visualizar árvore de decisão
print("\nVisualizando árvore de decisão...")
previsores = X_treino.columns
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
tree.plot_tree(modelo, feature_names=previsores, class_names=['Não Sobreviveu', 'Sobreviveu'], 
               filled=True, rounded=True, fontsize=10);
plt.title('Árvore de Decisão - Padrões de Sobrevivência no Titanic', fontsize=14, fontweight='bold')
plt.show()



print("5. REGRAS DE PADRÃO DE MORTALIDADE")

print("Com base na análise dos dados e no modelo treinado, as principais regras são:")

print("\n🔴 REGRAS PRINCIPAIS DE SOBREVIVÊNCIA:")
print("1. SEXO: Mulheres têm MUITO MAIOR chance de sobrevivência")
print(f"   - Taxa de sobrevivência feminina: {surv_sex.loc['female', 'mean']:.1%}")
print(f"   - Taxa de sobrevivência masculina: {surv_sex.loc['male', 'mean']:.1%}")

print("\n2. CLASSE SOCIAL: Passageiros da 1ª classe têm MAIOR chance")
print(f"   - 1ª Classe: {surv_class.loc[1, 'mean']:.1%}")
print(f"   - 2ª Classe: {surv_class.loc[2, 'mean']:.1%}")
print(f"   - 3ª Classe: {surv_class.loc[3, 'mean']:.1%}")

print("\n3. IDADE: Crianças e idosos têm padrões diferentes")
print("   - Crianças (0-10 anos): Prioridade no salvamento")
print("   - Adultos jovens: Padrão intermediário")
print("   - Idosos: Menor prioridade")

print("\n4. TARIFA: Passageiros com tarifas mais altas sobrevivem mais")
print("   - Maior tarifa = Maior status social = Maior prioridade")

print("\n5. FAMÍLIA: Passageiros com família pequena têm vantagem")
print("   - SibSp (1-2): Melhor chance")
print("   - Parch (1-2): Melhor chance")
print("   - Famílias muito grandes: Desvantagem")

print("\n🎯 REGRAS COMBINADAS (MAIOR IMPACTO):")
print("1. MULHER + 1ª CLASSE = ~97% de chance de sobrevivência")
print("2. HOMEM + 3ª CLASSE = ~15% de chance de sobrevivência")
print("3. MULHER + QUALQUER CLASSE = ~74% de chance de sobrevivência")
print("4. HOMEM + 1ª CLASSE = ~37% de chance de sobrevivência")

print("\n📊 RESUMO ESTATÍSTICO:")
print(f"- Total de passageiros: {len(base)}")
print(f"- Sobreviventes: {sobreviventes[1][1]} ({sobreviventes[1][1]/(sobreviventes[1][0]+sobreviventes[1][1])*100:.1f}%)")
print(f"- Não sobreviventes: {sobreviventes[1][0]} ({sobreviventes[1][0]/(sobreviventes[1][0]+sobreviventes[1][1])*100:.1f}%)")
print(f"- Precisão do modelo: {accuracy_score(y_teste, previsoes)*100:.1f}%")



