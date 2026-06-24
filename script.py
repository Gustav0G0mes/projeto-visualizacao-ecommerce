import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo dos gráficos
sns.set_theme(style="whitegrid")

# 1. Leitura do arquivo
df = pd.read_csv("ecommerce_estatistica.csv")

# Mostrar informações iniciais
print("Primeiras 5 linhas:")
print(df.head())

print("\nColunas do arquivo:")
print(df.columns.tolist())

print("\nInformações gerais:")
df.info()

print("\nEstatísticas descritivas:")
print(df.describe(include="all"))

# 2. Ajustes automáticos
possiveis_precos = ["Preco", "Preço", "preco", "preço"]
possiveis_descontos = ["Desconto", "desconto"]
possiveis_notas = ["Nota", "nota"]
possiveis_avaliacoes = ["N_Avaliacoes", "Avaliacoes", "Avaliações", "n_avaliacoes"]
possiveis_categoria = ["Temporada", "Categoria", "Marca", "Material", "temporada", "categoria"]

def encontrar_coluna(lista_possiveis, colunas_df):
    for col in lista_possiveis:
        if col in colunas_df:
            return col
    return None

col_preco = encontrar_coluna(possiveis_precos, df.columns)
col_desconto = encontrar_coluna(possiveis_descontos, df.columns)
col_nota = encontrar_coluna(possiveis_notas, df.columns)
col_avaliacoes = encontrar_coluna(possiveis_avaliacoes, df.columns)
col_categoria = encontrar_coluna(possiveis_categoria, df.columns)

print("\nColunas encontradas:")
print("Preço:", col_preco)
print("Desconto:", col_desconto)
print("Nota:", col_nota)
print("Avaliações:", col_avaliacoes)
print("Categoria:", col_categoria)

# 3. Tratamento básico
if col_desconto and df[col_desconto].dtype == object:
    df[col_desconto] = df[col_desconto].astype(str).str.extract(r"(\d+)")[0]
    df[col_desconto] = pd.to_numeric(df[col_desconto], errors="coerce")

# Converter colunas numéricas, se existirem
for col in [col_preco, col_desconto, col_nota, col_avaliacoes]:
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Histograma
if col_preco:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col_preco].dropna(), bins=20, edgecolor="black")
    plt.title("Histograma dos Preços dos Produtos")
    plt.xlabel("Preço")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

# 5. Dispersão
if col_desconto and col_preco:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[col_desconto], df[col_preco], alpha=0.7)
    plt.title("Relação entre Desconto e Preço")
    plt.xlabel("Desconto")
    plt.ylabel("Preço")
    plt.tight_layout()
    plt.show()

# 6. Mapa de calor
colunas_numericas = [col for col in [col_preco, col_desconto, col_nota, col_avaliacoes] if col is not None]

if len(colunas_numericas) >= 2:
    plt.figure(figsize=(8, 6))
    correlacao = df[colunas_numericas].corr()
    sns.heatmap(correlacao, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de Calor das Correlações")
    plt.tight_layout()
    plt.show()

# 7. Gráfico de barras
if col_categoria and col_preco:
    media_por_categoria = df.groupby(col_categoria)[col_preco].mean().sort_values()

    plt.figure(figsize=(10, 5))
    media_por_categoria.plot(kind="bar")
    plt.title("Preço Médio por Categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Preço Médio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 8. Gráfico de pizza
if col_categoria:
    contagem_categoria = df[col_categoria].value_counts().head(8)

    plt.figure(figsize=(8, 8))
    plt.pie(
        contagem_categoria,
        labels=contagem_categoria.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Distribuição por Categoria")
    plt.tight_layout()
    plt.show()

# 9. Gráfico de densidade
if col_preco:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df[col_preco].dropna(), fill=True)
    plt.title("Densidade dos Preços")
    plt.xlabel("Preço")
    plt.ylabel("Densidade")
    plt.tight_layout()
    plt.show()

# 10. Gráfico de regressão
if col_desconto and col_preco:
    plt.figure(figsize=(8, 5))
    sns.regplot(
        x=col_desconto,
        y=col_preco,
        data=df,
        scatter_kws={"alpha": 0.6}
    )
    plt.title("Regressão entre Desconto e Preço")
    plt.xlabel("Desconto")
    plt.ylabel("Preço")
    plt.tight_layout()
    plt.show()