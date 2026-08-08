<h1 align="center">Master Script — scikit-learn</h1>

<p align="center">
  Guia de consulta com pipelines e grades de hiperparâmetros das principais famílias de modelos.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn 1.9.0">
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/testes-48%2F48-4CAF50" alt="48 de 48 verificações">
</p>

---

## Índice

- [Sobre](#sobre)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Conteúdo](#conteúdo)
- [Como o código está organizado](#como-o-código-está-organizado)
- [Notas de implementação](#notas-de-implementação)
- [Ambiente validado](#ambiente-validado)
- [Estrutura de arquivos](#estrutura-de-arquivos)

---

## Sobre

Repositório pessoal de consulta: sintaxe, declaração de pipelines e **grades de
hiperparâmetros** das principais famílias matemáticas do scikit-learn, para consultar
sempre que um projeto novo começa.

O código é referência, não um projeto executável de ponta a ponta. Os dados são fictícios
e servem apenas para dar forma ao fluxo — a ideia é copiar o bloco necessário e trocar o
`read_csv`.

São três notebooks, um por tipo de problema, todos seguindo o mesmo fluxo de seis etapas:

| # | Etapa | Conteúdo |
|:-:|-------|----------|
| 0 | Setup | Imports canônicos, `RANDOM_STATE`, config global |
| 1 | Data Prep | Leitura, tipagem, duplicatas, faltantes, `X` / `y` |
| 2 | Split | Treino/teste, quando usar validação separada, variantes por grupo e temporal |
| 3 | Pipelines | `ColumnTransformer` + um pipeline por família matemática |
| 4 | Validação Cruzada | Torneio entre todas as famílias, escolha da vencedora |
| 5 | Otimização | `RandomizedSearchCV` + **grades de todas as famílias** |
| 6 | Prova Final | Métricas, matriz de confusão, diagnóstico, persistência |
| A | Apêndice | Cheatsheet de escolha de família, métrica e diagnóstico de sintomas |

---

## Requisitos

- [uv](https://docs.astral.sh/uv/) — gerencia Python, virtualenv e dependências
- Python 3.12 ou superior (o `uv` baixa a versão certa sozinho)

---

## Instalação

```bash
uv sync
```

O comando lê o `.python-version`, garante o Python 3.14, cria o `.venv/` e instala as
dependências nas versões exatas do `uv.lock`.

Pacotes opcionais (`shap`, `optuna`, `umap-learn`, `kneed`, `kmodes`, `openpyxl`,
`sqlalchemy`):

```bash
uv sync --all-extras
```

Passo a passo completo, extras, atualização e solução de problemas em
[INSTALACAO.md](INSTALACAO.md).

---

## Uso

```bash
uv run jupyter lab
```

Não é preciso ativar o virtualenv: `uv run` resolve o ambiente sozinho.

---

## Conteúdo

| Notebook | Escopo | Pipelines | Células |
|----------|--------|:---------:|:-------:|
| [`01_classificacao.ipynb`](01_classificacao.ipynb) | Alvo categórico | 20 famílias | 39 |
| [`02_regressao.ipynb`](02_regressao.ipynb) | Alvo contínuo | 25 famílias | 39 |
| [`03_clusterizacao.ipynb`](03_clusterizacao.ipynb) | Sem alvo | 15 famílias | 37 |

<details>
<summary><b>01 — Classificação</b> (20 famílias)</summary>

<br>

Logistic, Ridge, SGD, SVC (rbf/poly), LinearSVC, KNN, GaussianNB, LDA, QDA, DecisionTree,
RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, MLP, XGBoost,
LightGBM, CatBoost, Dummy.

Inclui ainda `VotingClassifier` e `StackingClassifier`, três estratégias para
desbalanceamento (`class_weight`, SMOTE via `imblearn`, ajuste de threshold), curvas
ROC/PR/calibração e importância por permutação.

</details>

<details>
<summary><b>02 — Regressão</b> (25 famílias)</summary>

<br>

OLS, Ridge, Lasso, ElasticNet, BayesianRidge, SGD, Huber, RANSAC, QuantileRegressor,
PolyRidge, Poisson GLM, Gamma GLM, SVR, LinearSVR, KernelRidge, KNN, DecisionTree,
RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, MLP, XGBoost,
LightGBM, CatBoost.

Inclui ainda `TransformedTargetRegressor` (alvo em log), PLS, GaussianProcess, predição
com intervalo por regressão quantílica, análise de resíduos e dependência parcial.

</details>

<details>
<summary><b>03 — Clusterização</b> (15 famílias)</summary>

<br>

KMeans, MiniBatchKMeans, BisectingKMeans, Agglomerative (ward e average), DBSCAN, HDBSCAN,
OPTICS, GaussianMixture, BayesianGMM, Spectral, MeanShift, Birch, AffinityPropagation.

Inclui ainda curva do cotovelo, curva k-distância para o `eps` do DBSCAN, dendrograma,
teste de estabilidade por reamostragem (ARI), diagrama de silhueta e perfil dos clusters.

</details>

---

## Como o código está organizado

**Fábrica de pré-processadores.** Em vez de repetir `ColumnTransformer` vinte vezes, cada
notebook tem `build_preprocessor(kind=...)` com receitas prontas:

| Receita | Numéricas | Categóricas | Para quem |
|---------|-----------|-------------|-----------|
| `"scaled"` | mediana + `StandardScaler` | moda + `OneHotEncoder` | Linear, SVM, KNN, MLP, LDA |
| `"tree"` | mediana, sem escala | moda + `OrdinalEncoder` | Árvore, RF, ExtraTrees, GB, AdaBoost |
| `"native"` | sem escala | `OrdinalEncoder` categórica nativa | HistGB, LightGBM, XGBoost, CatBoost |

**Convenção de nomes.** Todo pipeline termina no passo `"model"`. Por isso as grades usam
sempre o prefixo `model__`, e os blocos de pré-processamento usam `prep__num__...`.

**Dicionário `GRIDS`.** O coração do material: uma grade por família, escrita com
`loguniform` / `randint` / `uniform` do scipy — o formato que o `RandomizedSearchCV`
aproveita de verdade (a grade fixa do `GridSearchCV` não aceita distribuição contínua).

> [!IMPORTANT]
> Regra de ouro que atravessa os três documentos: tudo que *aprende* com os dados
> (imputação, escala, encoding, seleção de variável, balanceamento) mora **dentro** do
> Pipeline. Fora dele é vazamento de dados e métrica de validação inflada.

---

## Notas de implementação

**Clusterização não cabe no molde supervisionado, e o notebook diz isso.** Sem `y` não
existe acerto a medir. O split da etapa 2 muda de função: testa se a estrutura **se repete**
fora da amostra. E `DBSCAN`, `HDBSCAN`, `OPTICS`, `Agglomerative` e `Spectral` **não têm
`predict`** — então a etapa 5 documenta três rotas de busca (`RandomizedSearchCV` com CV
real, varredura sem divisão, e laço com `ParameterSampler`), explicando qual serve para cada
algoritmo. A validação principal é **estabilidade por reamostragem** (ARI entre execuções),
não a silhueta isolada.

**Em regressão, a transformação do alvo é etapa de primeira classe.** Alvo assimétrico
(preço, renda, contagem) costuma dar o maior ganho isolado do projeto. O notebook usa
`TransformedTargetRegressor` — que aplica `log1p` no fit e `expm1` no predict dentro da CV,
evitando o vazamento clássico de transformar o alvo antes do split — e traz uma célula que
roda o torneio inteiro com e sem log para comparar.

**Em classificação, o threshold ganhou célula própria.** 0.5 é o padrão, não o ótimo. São
três critérios: máximo F1, recall mínimo de negócio e custo esperado mínimo. Está anotado no
próprio notebook que escolher o corte olhando o teste já é uma forma leve de vazamento — o
certo é calibrar na validação.

---

## Ambiente validado

Testado nas versões abaixo, com **48 verificações** cobrindo as APIs que os notebooks usam
(pipelines, buscas, métricas, gráficos, os três boosters externos e as rotas de
clusterização).

| Componente | Versão |
|------------|--------|
| Python | 3.14.6 |
| scikit-learn | 1.9.0 |
| numpy · pandas · scipy | 2.4.6 · 3.0.5 · 1.18.0 |
| matplotlib | 3.11.1 |
| xgboost · lightgbm · catboost | 3.4.0 · 4.7.0 · 1.2.10 |
| imbalanced-learn | 0.14.2 |

> [!NOTE]
> Duas observações que vieram desse teste e valem para o seu código também:
>
> - **pandas 3.0** mudou comportamentos (Copy-on-Write por padrão, novo dtype de string).
>   Tudo que os notebooks usam foi verificado nessa versão, incluindo
>   `groupby.apply(include_groups=False)` e `make_column_selector` com colunas `category`.
> - **O truque `cv=[(slice(None), slice(None))]`**, comum em receitas de clusterização na
>   internet, **parou de funcionar com DataFrame** a partir do scikit-learn 1.7 — o
>   indexador interno passou a rejeitar `slice` para selecionar linhas. O notebook 03 usa
>   `[(np.arange(n), np.arange(n))]`, com a nota explicando o porquê.

---

## Estrutura de arquivos

```
.
├── 01_classificacao.ipynb    # 20 famílias, alvo categórico
├── 02_regressao.ipynb        # 25 famílias, alvo contínuo
├── 03_clusterizacao.ipynb    # 15 famílias, sem alvo
├── pyproject.toml            # dependências e extras opcionais
├── uv.lock                   # 139 pacotes com versão exata
├── .python-version           # 3.14
├── INSTALACAO.md             # guia de instalação
└── README.md
```
