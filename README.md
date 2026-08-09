<h1 align="center">Master Script — scikit-learn</h1>

<p align="center">
  Guia de estudo e consulta: cada família de modelos, seu pipeline e onde aplicar.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn 1.9.0">
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## Índice

**Começando**
- [Como usar este guia](#como-usar-este-guia)
- [Instalação e uso](#instalação-e-uso)

**Parte 1 — Fundamentos**
- [O Pipeline e por que ele é obrigatório](#o-pipeline-e-por-que-ele-é-obrigatório)
- [As três receitas de pré-processamento](#as-três-receitas-de-pré-processamento)
- [Escalonamento: qual e quando](#escalonamento-qual-e-quando)
- [Codificação de categóricas](#codificação-de-categóricas)
- [Imputação de faltantes](#imputação-de-faltantes)
- [Validação cruzada](#validação-cruzada)
- [Viés e variância](#viés-e-variância)

**Parte 2 — [Classificação](#parte-2--classificação)**
- [Lineares](#lineares) · [Margem máxima](#margem-máxima-svm) · [Distância](#distância-knn) · [Bayesiano](#bayesiano-naive-bayes) · [Discriminante](#discriminante-lda--qda)
- [Árvore](#árvore-de-decisão) · [Bagging](#ensembles-de-bagging) · [Boosting](#ensembles-de-boosting) · [Rede neural](#rede-neural-mlp) · [Meta-ensembles](#meta-ensembles)
- [Desbalanceamento](#desbalanceamento-de-classes) · [Métricas](#métricas-de-classificação)

**Parte 3 — [Regressão](#parte-3--regressão)**
- [Lineares](#lineares-regularizados) · [Robustos](#lineares-robustos) · [GLM](#glm-modelos-lineares-generalizados) · [Não-linearidade explícita](#não-linearidade-explícita)
- [Kernel](#métodos-de-kernel) · [Árvores e ensembles](#árvores-e-ensembles-em-regressão) · [Transformação do alvo](#transformação-do-alvo) · [Métricas](#métricas-de-regressão)

**Parte 4 — [Clusterização](#parte-4--clusterização)**
- [Centróide](#centróide-kmeans-e-variantes) · [Hierárquico](#hierárquico-agglomerative) · [Densidade](#densidade-dbscan-hdbscan-optics) · [Probabilístico](#probabilístico-gaussian-mixture)
- [Grafo](#grafo-spectral-clustering) · [Outros](#outras-famílias) · [Escolha do k](#como-escolher-o-número-de-clusters) · [Métricas](#métricas-de-clusterização)

**Parte 5 — Aplicando**
- [Otimização de hiperparâmetros](#parte-5--otimização-de-hiperparâmetros)
- [Guia rápido de decisão](#parte-6--guia-rápido-de-decisão)
- [Estrutura de arquivos](#estrutura-de-arquivos)

---

## Como usar este guia

Este documento explica **o que cada modelo faz, quando usar e o que ajustar**. Os
notebooks trazem o código pronto para copiar:

| Notebook | Escopo | Famílias |
|----------|--------|:--------:|
| [`01_classificacao.ipynb`](01_classificacao.ipynb) | Alvo categórico | 20 |
| [`02_regressao.ipynb`](02_regressao.ipynb) | Alvo contínuo | 25 |
| [`03_clusterizacao.ipynb`](03_clusterizacao.ipynb) | Sem alvo | 15 |

Os três seguem o mesmo fluxo de seis etapas: Data Prep → Split → Pipelines → Validação
Cruzada → Otimização → Prova Final.

Cada modelo aqui segue o mesmo formato: **o que faz** (a intuição matemática), **quando
usar**, **quando evitar**, **pré-processamento exigido**, **hiperparâmetros que movem a
agulha** e **aplicação típica**.

---

## Instalação e uso

```bash
uv sync
```

```bash
uv run jupyter lab
```

Extras opcionais, atualização e solução de problemas: [INSTALACAO.md](INSTALACAO.md).

---

# Parte 1 — Fundamentos

## O Pipeline e por que ele é obrigatório

Um `Pipeline` encadeia transformações e termina num estimador. Ele não é açúcar sintático:
é o que impede **vazamento de dados** (*data leakage*).

Considere imputar a média de uma coluna. Se você calcular a média sobre o dataset inteiro
e só depois separar treino e teste, a média já carrega informação das linhas de teste. O
modelo é avaliado sobre dados que influenciaram seu próprio pré-processamento — e a
métrica de validação sai inflada. Em produção, sem esse privilégio, o desempenho cai.

O Pipeline resolve porque `fit` e `transform` são separados:

```python
pipe = Pipeline([("prep", preprocessor), ("model", LogisticRegression())])
cross_val_score(pipe, X_train, y_train, cv=5)
```

Em cada fold, o `prep` é ajustado **só** nas linhas de treino daquele fold e apenas
aplicado nas de validação. Sem Pipeline, você teria que replicar isso à mão em cada fold —
e é aí que o erro acontece.

**O que precisa estar dentro do Pipeline:** imputação, escala, codificação, seleção de
variáveis, redução de dimensionalidade, balanceamento (SMOTE), transformação do alvo.
Qualquer coisa que *aprende parâmetros a partir dos dados*.

**O que pode ficar fora:** remoção de colunas identificadoras, correção de tipos, remoção
de duplicatas, filtros de regra de negócio. Operações que não estimam nada.

> [!IMPORTANT]
> Regra prática: se a operação tem um `.fit()`, ela pertence ao Pipeline.

## As três receitas de pré-processamento

Modelos diferentes exigem preparos diferentes. Em vez de repetir `ColumnTransformer`
dezenas de vezes, os notebooks usam `build_preprocessor(kind=...)`:

| Receita | Numéricas | Categóricas | Para quem |
|---------|-----------|-------------|-----------|
| `"scaled"` | mediana + `StandardScaler` | moda + `OneHotEncoder` | Linear, SVM, KNN, MLP, LDA/QDA |
| `"tree"` | mediana, **sem escala** | moda + `OrdinalEncoder` | Árvore, RF, ExtraTrees, GB, AdaBoost |
| `"native"` | sem escala | `OrdinalEncoder` marcada como categórica | HistGB, LightGBM, XGBoost, CatBoost |

A lógica por trás:

- **Modelos que medem distância ou usam gradiente** (linear, SVM, KNN, redes) precisam de
  escala. Sem ela, uma coluna em reais (0–100.000) domina outra em anos (0–80): a
  distância euclidiana vira essencialmente a diferença de renda.
- **Árvores são invariantes a transformações monotônicas.** Elas perguntam "renda > X?" —
  e o ponto de corte se ajusta sozinho, seja a escala qual for. Escalar não prejudica, mas
  é trabalho desperdiçado.
- **One-Hot em árvore com categórica de alta cardinalidade é ruim.** Cada categoria vira
  uma coluna binária esparsa; a árvore precisa gastar um nível inteiro para isolar cada
  uma, fragmentando os dados. `OrdinalEncoder` mantém uma coluna só e deixa a árvore
  encontrar os agrupamentos.

**Convenção de nomes.** Todo pipeline termina no passo `"model"`. Por isso as grades usam
sempre o prefixo `model__`, e os blocos de pré-processamento usam `prep__num__...`.

## Escalonamento: qual e quando

| Transformador | O que faz | Use quando |
|---------------|-----------|------------|
| `StandardScaler` | média 0, desvio 1 | Padrão. Dados aproximadamente simétricos |
| `RobustScaler` | usa mediana e IQR | Há outliers — a mediana não se move com eles |
| `MinMaxScaler` | comprime para [0, 1] | Redes neurais, ou quando o algoritmo assume domínio limitado |
| `PowerTransformer` | Yeo-Johnson: aproxima da normal | Assimetria forte (`skew > 1`) |
| `QuantileTransformer` | converte para rank, depois normal/uniforme | Distribuição muito irregular; é agressivo e destrói a forma original |
| `Normalizer` | norma L2 **por linha** | Quando o que importa é a direção do vetor, não a magnitude (texto, sinais) |

Atenção: `Normalizer` opera em linhas, todos os outros em colunas. É a confusão mais
comum da lista.

## Codificação de categóricas

| Codificador | Saída | Use quando |
|-------------|-------|------------|
| `OneHotEncoder` | uma coluna binária por categoria | Modelo linear, SVM, KNN, rede neural |
| `OrdinalEncoder` | um inteiro por categoria | Árvores e ensembles baseados nelas |
| `TargetEncoder` | média do alvo por categoria | Alta cardinalidade (CEP, produto, cidade) |

Três parâmetros que evitam quebra em produção:

- `handle_unknown="infrequent_if_exist"` (One-Hot): categoria nova no teste cai no balde
  "infrequente" em vez de gerar erro.
- `min_frequency=0.01` (One-Hot): agrupa categorias raras, reduzindo a explosão de colunas.
- `handle_unknown="use_encoded_value", unknown_value=-1` (Ordinal): categoria nova vira −1.

**`TargetEncoder` merece cuidado.** Ele usa o alvo para construir a variável, o que é
vazamento por construção. A implementação do scikit-learn faz validação cruzada interna
para mitigar — mas ainda assim, use **sempre dentro do Pipeline**, nunca antes do split.

**Categórica ordinal de verdade** (`baixo < médio < alto`) merece codificação manual com a
ordem correta, não One-Hot — que joga fora a informação de ordem.

## Imputação de faltantes

| Estratégia | Quando |
|------------|--------|
| `SimpleImputer(strategy="median")` | Padrão para numéricas — robusta a outliers |
| `SimpleImputer(strategy="most_frequent")` | Padrão para categóricas |
| `SimpleImputer(strategy="constant", fill_value="AUSENTE")` | Quando "ausente" é informação, não ruído |
| `KNNImputer` | Poucas colunas, faltantes correlacionados entre si |
| Nativo do modelo | `HistGradientBoosting`, `LightGBM`, `XGBoost` tratam `NaN` sozinhos |

Antes de escolher, pergunte **por que** o dado falta. Se "renda ausente" acontece
sobretudo em inadimplentes, a ausência é sinal — imputar a mediana apaga esse sinal. Nesse
caso, adicione uma coluna indicadora:

```python
SimpleImputer(strategy="median", add_indicator=True)
```

Nunca impute o **alvo**. Linha sem `y` não serve para aprendizado supervisionado: remova.

## Validação cruzada

Dividir treino/teste uma vez dá **uma** estimativa, sujeita ao acaso daquela divisão. A
validação cruzada divide o treino em *k* partes, treina *k* vezes e devolve *k* medidas —
com média **e desvio**.

| Estratégia | Quando usar |
|------------|-------------|
| `KFold` | Regressão, dados independentes |
| `StratifiedKFold` | **Classificação, sempre** — preserva a proporção das classes |
| `RepeatedKFold` / `RepeatedStratifiedKFold` | Dataset pequeno: repete com embaralhamentos diferentes |
| `GroupKFold` / `StratifiedGroupKFold` | Linhas repetidas por entidade (cliente, paciente) |
| `TimeSeriesSplit` | Ordem temporal — só passado prevê futuro |

> [!WARNING]
> Diferença menor que o desvio-padrão entre folds **não é diferença**. Um modelo com
> 0,842 ± 0,03 e outro com 0,838 ± 0,03 estão empatados. Desempate pelo mais simples ou
> mais rápido.

**Os dois erros que invalidam tudo:** embaralhar série temporal (o modelo aprende o futuro)
e deixar a mesma entidade nos dois lados do split (o modelo decora o cliente em vez de
aprender o padrão).

## Viés e variância

Todo erro de um modelo se decompõe em três partes: **viés** (o modelo é simples demais
para o padrão real), **variância** (o modelo é sensível demais à amostra específica) e
**ruído irredutível**.

| Sintoma na CV | Diagnóstico | Correção |
|---------------|-------------|----------|
| Treino ruim, validação ruim (parecidos) | Viés alto — *underfitting* | Mais capacidade: modelo mais complexo, mais variáveis, interações |
| Treino ótimo, validação ruim | Variância alta — *overfitting* | Mais regularização, menos profundidade, mais dados, ensemble |
| Treino ≈ validação, ambos bons | Equilibrado | Explore ganhos em variáveis, não em modelo |

É por isso que `cross_validate(..., return_train_score=True)` aparece nos notebooks: sem a
métrica de treino ao lado da de validação, você não sabe qual dos dois problemas está
enfrentando.

**Bagging reduz variância** (várias árvores independentes, média dos votos). **Boosting
reduz viés** (árvores sequenciais, cada uma corrigindo o erro da anterior). Saber disso
orienta a escolha: modelo instável pede bagging; modelo fraco pede boosting.

---

# Parte 2 — Classificação

> Notebook: [`01_classificacao.ipynb`](01_classificacao.ipynb)

## Lineares

Traçam uma **fronteira de decisão linear** no espaço das variáveis. Rápidos,
interpretáveis via coeficientes, e a referência contra a qual todo modelo complexo deve
justificar sua complexidade.

### LogisticRegression

**O que faz.** Modela a probabilidade da classe positiva como
`sigmoid(w·x + b)`. Apesar do nome, é classificação, não regressão. Otimiza a
*log-loss*, o que a torna naturalmente calibrada — as probabilidades que ela devolve
significam o que dizem.

**Quando usar.** Baseline obrigatório de qualquer projeto de classificação. Quando você
precisa explicar a decisão (setor regulado, crédito, saúde). Quando há muitas colunas e
poucas linhas. Quando a probabilidade importa mais que o rótulo.

**Evite quando.** A fronteira real é claramente não-linear e você não quer construir
interações à mão.

**Pré-processamento.** `"scaled"` — obrigatório. A penalização L1/L2 age sobre a magnitude
dos coeficientes; sem escala comum, a regularização pune variáveis arbitrariamente.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `C` | **Inverso** da regularização. Menor = mais regularizado | `loguniform(1e-4, 1e3)` |
| `penalty` | `l2` encolhe; `l1` zera coeficientes (seleção); `elasticnet` mistura | — |
| `solver` | `liblinear`/`saga` aceitam L1; `lbfgs` só L2 | — |
| `class_weight` | `"balanced"` compensa desbalanceamento | — |

**Aplicação típica.** Score de crédito, propensão de compra, diagnóstico com probabilidade
auditável, previsão de churn quando o time precisa entender o porquê.

### RidgeClassifier

**O que faz.** Aplica regressão Ridge sobre o alvo codificado em ±1 e classifica pelo
sinal. Tem solução fechada — é notavelmente mais rápido que a logística.

**Quando usar.** Muitas variáveis correlacionadas entre si, quando você quer a velocidade e
não precisa de probabilidade.

**Evite quando.** Precisa de `predict_proba` — ele não tem. Só `decision_function`.

**Pré-processamento.** `"scaled"`.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `alpha` | Força da regularização L2. **Maior = mais regularizado** (inverso do `C`) | `loguniform(1e-3, 1e3)` |

**Aplicação típica.** Classificação de texto com muitas features, triagem rápida em
pipelines de alto volume.

### SGDClassifier

**O que faz.** Ajusta um modelo linear por descida de gradiente estocástica — uma amostra
por vez. Com `loss="log_loss"` é logística; com `loss="hinge"`, é SVM linear.

**Quando usar.** Dados que não cabem na memória, ou milhões de linhas onde a logística
comum fica lenta. Suporta aprendizado incremental via `partial_fit`.

**Evite quando.** O dataset é pequeno — a solução exata dos outros métodos é melhor e igualmente rápida.

**Pré-processamento.** `"scaled"` — crítico. SGD é muito sensível a escala; sem ela a
convergência fica lenta ou não acontece.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `loss` | Define qual modelo você está treinando | `log_loss`, `hinge`, `modified_huber` |
| `alpha` | Regularização | `loguniform(1e-6, 1e-1)` |
| `learning_rate` + `eta0` | Como o passo evolui | `optimal`, `adaptive` |

**Aplicação típica.** Classificação de texto em larga escala, sistemas com fluxo contínuo
de dados novos.

## Margem máxima (SVM)

### SVC

**O que faz.** Encontra o hiperplano que **maximiza a margem** entre as classes. O truque
do kernel projeta os dados num espaço de dimensão maior, onde uma fronteira linear
corresponde a uma fronteira curva no espaço original — sem nunca calcular essa projeção
explicitamente.

**Quando usar.** Dataset pequeno ou médio (até ~50 mil linhas), limpo, com fronteira
genuinamente não-linear. Espaço de alta dimensão — SVM lida bem com mais colunas que linhas.

**Evite quando.** Muitos dados: o custo é entre O(n²) e O(n³). Acima de ~100 mil linhas
torna-se inviável. Também sofre com ruído e outliers, que viram vetores de suporte.

**Pré-processamento.** `"scaled"` — absolutamente crítico. O kernel RBF calcula distâncias;
sem escala o resultado é lixo.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `C` | Rigidez da margem. Alto = tolera menos erro, arrisca overfitting | `loguniform(1e-2, 1e3)` |
| `gamma` | Alcance de cada ponto no kernel RBF. Alto = fronteira muito flexível | `loguniform(1e-5, 1e1)` |
| `kernel` | `rbf` é o padrão; `poly` para interações; `linear` para alta dimensão | — |

`C` e `gamma` interagem fortemente — otimize sempre os dois juntos.

> [!NOTE]
> `probability=True` faz o SVC rodar validação cruzada interna (Platt scaling) para
> produzir probabilidades. Custa ~5× mais tempo de treino. Só ative se precisar.

**Aplicação típica.** Classificação de imagens com poucos exemplos, bioinformática
(genes >> amostras), detecção de defeitos em manufatura.

### LinearSVC

**O que faz.** SVM com kernel linear, implementado por um otimizador especializado
(liblinear). Muito mais rápido que `SVC(kernel="linear")` para o mesmo resultado.

**Quando usar.** Texto vetorizado com TF-IDF — é um dos melhores baselines nesse domínio.
Alta dimensão em geral.

**Pré-processamento.** `"scaled"`.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `C` | Rigidez da margem | `loguniform(1e-3, 1e2)` |
| `loss` | `squared_hinge` (padrão) ou `hinge` | — |

**Aplicação típica.** Classificação de documentos, análise de sentimento, filtro de spam.

## Distância (KNN)

### KNeighborsClassifier

**O que faz.** Não treina nada. Guarda os dados e, na predição, encontra os *k* vizinhos
mais próximos e faz a votação. É o exemplo canônico de *lazy learning*.

**Quando usar.** Fronteira de decisão muito irregular, dados de baixa dimensão (até ~10-15
colunas), como baseline não-paramétrico rápido de montar.

**Evite quando.** Alta dimensão — vítima direta da **maldição da dimensionalidade**: acima
de ~15 colunas, todos os pontos ficam aproximadamente equidistantes e o conceito de
"vizinho" perde sentido. Também é lento na predição (busca em toda a base) e exige manter
o dataset inteiro em memória.

**Pré-processamento.** `"scaled"` — crítico, por definição.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `n_neighbors` | Poucos = fronteira ruidosa; muitos = fronteira suave demais | `randint(1, 50)` |
| `weights` | `"distance"` dá mais peso ao vizinho mais próximo | `uniform`, `distance` |
| `metric` | `manhattan` costuma se sair melhor em dimensão mais alta | — |

**Aplicação típica.** Sistemas de recomendação simples, imputação de dados, reconhecimento
de padrões em séries curtas.

## Bayesiano (Naive Bayes)

### GaussianNB e variantes

**O que faz.** Aplica o teorema de Bayes assumindo **independência condicional** entre as
variáveis — daí o "naive". A suposição é quase sempre falsa, e o modelo funciona bem mesmo
assim, porque para *classificar* basta acertar qual classe tem maior probabilidade, não o
valor exato dela.

**Variantes.** `GaussianNB` para numéricas contínuas; `MultinomialNB` para contagens
(texto); `ComplementNB` para texto desbalanceado; `BernoulliNB` para binárias.

**Quando usar.** Baseline instantâneo — treina em uma passada pelos dados. Texto. Poucos
dados de treino. Quando você precisa de algo funcionando em minutos.

**Evite quando.** As variáveis são fortemente correlacionadas (a suposição quebra de
verdade) ou quando você precisa de probabilidades calibradas — NB tende a produzir
probabilidades extremas, próximas de 0 e 1.

**Pré-processamento.** `"scaled"` para o `GaussianNB`. `MultinomialNB` exige valores não
negativos — use `MinMaxScaler` ou contagens brutas.

| Hiperparâmetro | Efeito |
|----------------|--------|
| `var_smoothing` (Gaussian) | Estabilidade numérica; raramente muda muito |
| `alpha` (Multinomial/Complement) | Suavização de Laplace para categorias não vistas |

**Aplicação típica.** Filtro de spam (o caso clássico), categorização de artigos,
diagnóstico preliminar.

## Discriminante (LDA / QDA)

### LinearDiscriminantAnalysis

**O que faz.** Assume que cada classe segue uma distribuição normal multivariada com a
**mesma matriz de covariância**. Sob essa hipótese, a fronteira ótima é linear. Também
funciona como técnica de **redução de dimensionalidade supervisionada** — projeta os dados
maximizando a separação entre classes.

**Quando usar.** Classes aproximadamente normais, dataset pequeno (é muito eficiente em
poucos dados), ou quando você quer reduzir dimensão levando o alvo em conta (diferente do
PCA, que ignora `y`).

**Pré-processamento.** `"scaled"`.

| Hiperparâmetro | Efeito |
|----------------|--------|
| `solver` | `svd` (padrão), `lsqr`, `eigen` |
| `shrinkage` | `"auto"` regulariza a covariância — essencial quando há mais colunas que linhas |

### QuadraticDiscriminantAnalysis

**O que faz.** Mesma ideia, mas cada classe tem sua **própria** matriz de covariância — o
que produz fronteira quadrática (curva).

**Quando usar.** Quando as classes têm dispersões visivelmente diferentes.

**Evite quando.** Poucos dados por classe: estimar uma matriz de covariância completa por
classe consome muitos graus de liberdade.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `reg_param` | Regulariza a covariância; suba se der erro de matriz singular | `uniform(0, 1)` |

**Aplicação típica.** Classificação de espécies por medidas morfológicas, controle de
qualidade com poucas amostras, reconhecimento de fala clássico.

## Árvore de decisão

### DecisionTreeClassifier

**O que faz.** Divide o espaço recursivamente com perguntas do tipo "variável > valor?",
escolhendo em cada nó o corte que mais reduz a impureza (Gini ou entropia). O resultado é
um conjunto de regras legíveis.

**Quando usar.** Quando a explicabilidade é o requisito principal — é o único modelo que
você pode desenhar num slide e um não-técnico entende. Também captura interações e
não-linearidades automaticamente.

**Evite quando.** Precisão importa. Uma árvore isolada tem **variância altíssima**: mudar
poucas linhas do treino pode gerar uma árvore completamente diferente. Na prática, use RF
ou boosting — que são árvores em conjunto justamente para corrigir isso.

**Pré-processamento.** `"tree"` — não precisa de escala.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `max_depth` | Principal controle de complexidade. `None` = cresce até a pureza total | `[None, 3, 5, 8, 12, 20]` |
| `min_samples_leaf` | Mínimo de amostras numa folha — suaviza a fronteira | `randint(1, 30)` |
| `min_samples_split` | Mínimo para tentar dividir um nó | `randint(2, 40)` |
| `ccp_alpha` | Poda por complexidade de custo, depois do crescimento | `loguniform(1e-5, 1e-1)` |
| `criterion` | `gini` (rápido) ou `entropy` (levemente mais equilibrado) | — |

**Aplicação típica.** Árvore de decisão de negócio, definição de regras de elegibilidade,
material didático, análise exploratória de quais variáveis separam os grupos.

## Ensembles de bagging

Treinam muitas árvores **em paralelo**, cada uma numa amostra diferente dos dados
(*bootstrap*) e de colunas. A média dos votos cancela os erros individuais — **reduzem
variância**.

### RandomForest

**O que faz.** Muitas árvores profundas, cada uma vendo um subconjunto aleatório de linhas
e, em cada nó, um subconjunto aleatório de colunas. A dupla aleatoriedade descorrelaciona
as árvores, e é a descorrelação que faz a média funcionar.

**Quando usar.** Segundo baseline depois da logística. Funciona bem sem ajuste nenhum,
lida com variáveis em escalas diferentes, é robusto a outliers e dá importância de
variáveis de graça. Excelente quando você tem pouco tempo.

**Evite quando.** Precisa extrapolar (árvores nunca preveem fora da faixa vista), o modelo
precisa ser pequeno (uma floresta de 500 árvores é grande), ou latência de predição é
crítica.

**Pré-processamento.** `"tree"`.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `n_estimators` | Mais árvores nunca piora — só custa tempo. 300-500 costuma saturar | `randint(200, 1200)` |
| `max_features` | **O parâmetro mais importante.** Quantas colunas considerar por nó. Menor = mais diversidade | `["sqrt", "log2", 0.3, 0.5, None]` |
| `max_depth` | `None` é o padrão e funciona bem em floresta | `[None, 5, 10, 20, 30]` |
| `min_samples_leaf` | Suaviza e reduz o tamanho do modelo | `randint(1, 15)` |
| `class_weight` | `"balanced_subsample"` recalcula o peso a cada bootstrap | — |

> [!TIP]
> A importância nativa (`feature_importances_`) é enviesada a favor de variáveis de alta
> cardinalidade. Para uma medida honesta, use `permutation_importance`.

**Aplicação típica.** Praticamente qualquer problema tabular como referência sólida:
risco de crédito, detecção de fraude, previsão de churn, priorização de leads.

### ExtraTrees

**O que faz.** Como o RandomForest, mas os pontos de corte são sorteados aleatoriamente em
vez de otimizados. Mais aleatoriedade ainda.

**Quando usar.** Quando o RandomForest está com variância alta. Treina mais rápido (não
procura o melhor corte) e às vezes generaliza melhor. Vale sempre testar os dois — custa
uma linha.

**Pré-processamento.** `"tree"`. Hiperparâmetros idênticos ao RandomForest.

## Ensembles de boosting

Treinam árvores **em sequência**, cada uma corrigindo os erros da anterior. **Reduzem
viés** e são, hoje, o estado da arte em dados tabulares.

### GradientBoosting

**O que faz.** Ajusta cada nova árvore ao **gradiente da função de perda** — ou seja, ao
resíduo do modelo até então. A implementação original do scikit-learn.

**Quando usar.** Datasets pequenos onde a precisão importa mais que o tempo de treino.

**Evite quando.** Mais de ~10 mil linhas — é sequencial e não paraleliza. Prefira
`HistGradientBoosting` ou LightGBM.

**Pré-processamento.** `"tree"`.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `learning_rate` | Contribuição de cada árvore. **Troca com `n_estimators`**: metade da taxa pede o dobro de árvores | `loguniform(1e-3, 0.3)` |
| `n_estimators` | Número de árvores | `randint(100, 800)` |
| `max_depth` | Em boosting, árvores **rasas** (2 a 8). Cada uma é um aprendiz fraco | `randint(2, 8)` |
| `subsample` | `< 1` amostra linhas por árvore — vira *stochastic gradient boosting* | `uniform(0.6, 0.4)` |

### HistGradientBoosting

**O que faz.** Discretiza as variáveis contínuas em *bins* (padrão: 255) antes de procurar
os cortes. Isso transforma a busca de O(n) em O(bins) e acelera drasticamente. É a resposta
do scikit-learn ao LightGBM — e o algoritmo foi inspirado nele.

**Quando usar.** **Primeira escolha para dados tabulares.** Rápido, preciso, trata `NaN`
nativamente, suporta categóricas sem codificação e tem *early stopping* embutido.

**Pré-processamento.** `"native"` — informe `categorical_features` e deixe os `NaN` passar.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `learning_rate` | Como no GB | `loguniform(1e-3, 0.3)` |
| `max_iter` | Número de árvores (equivale a `n_estimators`) | `randint(100, 800)` |
| `max_leaf_nodes` | **Principal controle de complexidade** aqui, mais que `max_depth` | `randint(10, 128)` |
| `l2_regularization` | Penaliza folhas com valores extremos | `loguniform(1e-6, 1e1)` |
| `early_stopping` | Para quando a validação interna deixa de melhorar | `True` |

**Aplicação típica.** O modelo que você usa quando quer o melhor resultado com esforço
razoável: previsão de demanda, precificação, risco, ranking.

### AdaBoost

**O que faz.** O boosting original (1995). Aumenta o peso das amostras classificadas
erradas a cada rodada, forçando as árvores seguintes a se concentrarem nelas.

**Quando usar.** Valor sobretudo didático hoje — é o mais simples de explicar. Funciona
bem com *stumps* (árvores de profundidade 1).

**Evite quando.** Há ruído ou outliers: ao aumentar o peso dos erros, o AdaBoost persegue
justamente os pontos ruidosos.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `n_estimators` | Número de aprendizes fracos | `randint(50, 600)` |
| `learning_rate` | Contribuição de cada um | `loguniform(1e-3, 2.0)` |
| `estimator` | O aprendiz fraco — normalmente `DecisionTree(max_depth=1..3)` | — |

### XGBoost

**O que faz.** Boosting com regularização explícita no objetivo (L1 e L2 sobre os pesos
das folhas), tratamento nativo de faltantes e implementação altamente otimizada. Foi o
algoritmo que dominou competições de dados tabulares por anos.

**Quando usar.** Quando precisa do último ponto percentual de desempenho e tem tempo para
otimizar. Ecossistema maduro, boa documentação, integração com SHAP.

**Pré-processamento.** `"native"`.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `learning_rate` | Troca com `n_estimators` | `loguniform(1e-3, 0.3)` |
| `max_depth` | Complexidade por árvore | `randint(2, 12)` |
| `min_child_weight` | Peso mínimo (hessiana) por folha — regularizador forte | `randint(1, 20)` |
| `subsample` / `colsample_bytree` | Amostragem de linhas / colunas | `uniform(0.5, 0.5)` |
| `gamma` | Ganho mínimo para justificar um corte | `loguniform(1e-4, 5)` |
| `reg_alpha` / `reg_lambda` | L1 / L2 nas folhas | `loguniform(...)` |
| `scale_pos_weight` | `n_neg / n_pos` para desbalanceamento | — |

**Aplicação típica.** Competições, sistemas de ranking, scoring de risco, detecção de
fraude em escala.

### LightGBM

**O que faz.** Cresce as árvores **por folha** (*leaf-wise*), escolhendo sempre a folha com
maior ganho — em vez de nível por nível. Produz árvores mais profundas e assimétricas, e
converge com menos iterações.

**Quando usar.** Datasets grandes (centenas de milhares a milhões de linhas) onde a
velocidade importa. Excelente com categóricas de alta cardinalidade.

**Cuidado.** O crescimento *leaf-wise* superajusta com facilidade em datasets pequenos.
Controle com `num_leaves` e `min_child_samples`.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `num_leaves` | **O principal.** Regra prática: mantenha `< 2^max_depth` | `randint(15, 255)` |
| `min_child_samples` | Mínimo de amostras por folha — trave do overfitting | `randint(5, 100)` |
| `learning_rate` | Como sempre | `loguniform(1e-3, 0.3)` |
| `subsample` + `subsample_freq` | Amostragem de linhas — **`subsample` só age se `freq > 0`** | — |

**Aplicação típica.** Sistemas de recomendação, *click-through rate*, previsão em larga
escala, qualquer coisa com milhões de linhas.

### CatBoost

**O que faz.** Trata categóricas nativamente com *target encoding ordenado* (usa apenas as
linhas anteriores para calcular a estatística, evitando vazamento) e usa árvores
*oblivious* — o mesmo corte em todos os nós de um nível, o que regulariza e acelera a
predição.

**Quando usar.** Muitas variáveis categóricas, especialmente de alta cardinalidade. É o
que dá melhor resultado sem ajuste — os padrões são bem escolhidos.

**Evite quando.** Todas as variáveis são numéricas e a velocidade de treino importa: aí
LightGBM costuma ser mais rápido.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `depth` | Profundidade (uniforme, por ser *oblivious*) | `randint(4, 10)` |
| `l2_leaf_reg` | Regularização L2 | `loguniform(1e-1, 30)` |
| `learning_rate` | Como sempre | `loguniform(1e-3, 0.3)` |
| `auto_class_weights` | `"Balanced"` para desbalanceamento | — |

**Aplicação típica.** Dados de e-commerce e marketing (cheios de categóricas), previsão
com variáveis de localização, segmentação com IDs de produto.

## Rede neural (MLP)

### MLPClassifier

**O que faz.** Camadas de neurônios totalmente conectadas com ativação não-linear. Aprende
representações intermediárias em vez de trabalhar direto nas variáveis originais.

**Quando usar.** Muitos dados (dezenas de milhares para cima), padrões genuinamente
complexos, ou quando você já sabe que redes funcionam naquele domínio.

**Evite quando.** Dados tabulares comuns. Este é um ponto importante: **em tabular,
boosting quase sempre vence redes neurais**, treina mais rápido e exige menos ajuste. Use
MLP quando tiver motivo específico, não por reflexo.

**Pré-processamento.** `"scaled"`, preferencialmente `MinMaxScaler` — redes são muito
sensíveis à escala de entrada.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `hidden_layer_sizes` | Arquitetura. Comece com uma camada; só aprofunde com motivo | `[(50,), (100,), (100, 50)]` |
| `alpha` | Regularização L2 | `loguniform(1e-6, 1e-1)` |
| `learning_rate_init` | Passo inicial do otimizador | `loguniform(1e-4, 1e-1)` |
| `early_stopping` | **Sempre `True`** — separa validação interna e para no melhor ponto | — |

**Aplicação típica.** Sinais e séries com padrão complexo, dados de sensores,
características extraídas de imagem ou áudio.

## Meta-ensembles

### VotingClassifier

**O que faz.** Combina modelos diferentes por votação. `voting="hard"` conta rótulos;
`voting="soft"` faz a média das probabilidades — quase sempre melhor, porque usa a
confiança de cada modelo.

**Quando usar.** Quando modelos de famílias diferentes erram em casos diferentes. Combinar
uma logística com um boosting costuma somar; combinar dois boostings parecidos, não.

### StackingClassifier

**O que faz.** Treina um **meta-modelo** sobre as previsões dos modelos de base. As
previsões usadas no treino do meta-modelo vêm de validação cruzada interna (`cv=5`), o que
evita vazamento.

**Quando usar.** Última etapa de otimização, quando você já esgotou os ganhos individuais.
Ganho típico é pequeno — na casa de 1-2%.

**Cuidado.** Custo de treino multiplicado, modelo final difícil de explicar e de colocar em
produção. Use `final_estimator=LogisticRegression()` — meta-modelo complexo superajusta.

## Desbalanceamento de classes

Quando uma classe representa 1-5% dos dados, a acurácia deixa de significar algo: prever
sempre "não" já acerta 95%. Três estratégias, em ordem de preferência:

**1. Peso de classe.** `class_weight="balanced"` faz o modelo pagar mais caro pelo erro na
classe rara. Não inventa dados, custo computacional zero. **Comece sempre por aqui.**

```python
LogisticRegression(class_weight="balanced")
RandomForestClassifier(class_weight="balanced_subsample")
XGBClassifier(scale_pos_weight=n_neg/n_pos)
```

**2. Ajuste do threshold.** Treine normalmente e escolha o corte pela curva
Precisão-Revocação. Frequentemente é a melhor relação custo-benefício — nenhum
retreinamento envolvido.

**3. Reamostragem.** SMOTE gera exemplos sintéticos da classe minoritária interpolando
vizinhos. Precisa do `ImbPipeline` do `imbalanced-learn`, porque a reamostragem só pode
agir no treino de cada fold:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
ImbPipeline([("prep", prep), ("smote", SMOTE()), ("model", modelo)])
```

> [!CAUTION]
> Aplicar SMOTE antes do split ou antes da validação cruzada é o erro mais comum da área.
> Exemplos sintéticos vazam para a validação e a métrica sobe artificialmente — às vezes
> 10 pontos ou mais.

## Métricas de classificação

| Métrica | O que mede | Use quando |
|---------|------------|------------|
| `accuracy` | Fração de acertos | Classes equilibradas, erros de custo igual |
| `balanced_accuracy` | Média do recall por classe | Desbalanceamento moderado |
| `precision` | Dos que previ positivo, quantos eram | Falso positivo é caro (spam, bloqueio de cartão) |
| `recall` | Dos positivos reais, quantos peguei | Falso negativo é caro (câncer, fraude) |
| `f1` | Média harmônica de precisão e recall | Equilíbrio entre os dois |
| `fbeta(beta=2)` | F com recall pesando 2× | Recall é prioridade explícita |
| `roc_auc` | Qualidade do **ranqueamento** | Vai escolher o threshold depois |
| `average_precision` | Área sob a curva PR | Desbalanceamento forte (< 5% de positivos) |
| `log_loss` | Qualidade da **probabilidade** | A probabilidade é o produto (preço, risco) |
| `matthews_corrcoef` | Correlação usando as 4 células da matriz | Métrica única e honesta em desbalanceamento |

**ROC-AUC vs. Average Precision.** Sob desbalanceamento forte, a ROC parece otimista
porque a taxa de falso positivo tem um denominador enorme (todos os negativos). A curva PR
não tem esse problema. Com 1% de positivos, prefira `average_precision`.

**A matriz de confusão continua sendo a leitura mais informativa.** Ela mostra *como* o
modelo erra, não só quanto. Um modelo com 95% de acurácia que nunca acerta a classe rara é
imediatamente visível ali.

---

# Parte 3 — Regressão

> Notebook: [`02_regressao.ipynb`](02_regressao.ipynb)

Três diferenças estruturais em relação à classificação:

1. **Não existe estratificação** — o `KFold` é embaralhado, não estratificado.
2. **O alvo também é pré-processado.** Alvo assimétrico pede `log1p` — costuma ser o ganho
   isolado mais alto do projeto.
3. **A análise de resíduos vale mais que a métrica.** Um R² de 0,90 com resíduo em forma de
   funil é um modelo quebrado.

## Lineares regularizados

### LinearRegression (OLS)

**O que faz.** Mínimos quadrados: encontra os coeficientes que minimizam a soma dos
quadrados dos erros. Solução analítica, sem hiperparâmetro nenhum.

**Quando usar.** Referência interpretável. Quando o número de linhas é muito maior que o de
colunas e não há colinearidade.

**Evite quando.** Variáveis correlacionadas entre si — os coeficientes ficam instáveis e
sem sentido interpretativo. É exatamente o problema que o Ridge resolve.

**Pré-processamento.** `"scaled"` (não é obrigatório para o ajuste, mas torna os
coeficientes comparáveis entre si).

### Ridge (L2)

**O que faz.** OLS com penalização sobre a **soma dos quadrados** dos coeficientes. Encolhe
todos em direção a zero, sem zerar nenhum.

**Quando usar.** Quase sempre preferível ao OLS puro. Trata colinearidade, estabiliza os
coeficientes, e o custo é um único hiperparâmetro.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `alpha` | Força da penalização. **Maior = mais regularizado** | `loguniform(1e-4, 1e4)` |

> [!TIP]
> `RidgeCV(alphas=np.logspace(-4, 4, 100))` encontra o `alpha` por validação cruzada
> interna, muito mais barato que uma busca externa.

**Aplicação típica.** Precificação, previsão de demanda com variáveis correlacionadas,
qualquer modelo que precise ser auditável.

### Lasso (L1)

**O que faz.** Penaliza a **soma dos valores absolutos**. A geometria dessa penalização faz
coeficientes chegarem exatamente a zero — ou seja, **seleção automática de variáveis**.

**Quando usar.** Muitas colunas, e você suspeita que poucas importam. Quando quer um modelo
esparso e enxuto.

**Evite quando.** Há grupos de variáveis correlacionadas: o Lasso escolhe uma
arbitrariamente e zera as demais, o que é instável. Prefira ElasticNet.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `alpha` | Maior = mais coeficientes zerados | `loguniform(1e-4, 1e2)` |

### ElasticNet

**O que faz.** Combina L1 e L2. Herda a seleção do Lasso e a estabilidade do Ridge.

**Quando usar.** O padrão quando há muitas variáveis **e** correlação entre elas — que é o
caso mais comum na prática.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `alpha` | Força total | `loguniform(1e-4, 1e2)` |
| `l1_ratio` | 0 = Ridge puro, 1 = Lasso puro | `uniform(0, 1)` |

**Aplicação típica.** Genômica, dados de sensores com muitos canais correlacionados,
modelos econômicos com muitos indicadores.

### BayesianRidge

**O que faz.** Trata os coeficientes como distribuições e estima a regularização a partir
dos próprios dados, em vez de exigir busca. Devolve **desvio-padrão da predição**.

**Quando usar.** Poucos dados, ou quando você precisa de incerteza junto com a previsão e
não quer otimizar `alpha`.

```python
y_pred, y_std = modelo.predict(X_test, return_std=True)
```

## Lineares robustos

Regressão comum minimiza o erro **quadrático** — o que faz um outlier com resíduo 10× maior
pesar 100× mais. Estes três lidam com isso.

### HuberRegressor

**O que faz.** Perda quadrática para resíduos pequenos, **linear** para grandes. Outliers
influenciam, mas não dominam.

**Quando usar.** Há outliers no alvo, mas eles são dados legítimos que você não quer
descartar.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `epsilon` | Onde a perda vira linear. Menor = mais robusto (1.35 é o padrão) | `uniform(1.0, 2.5)` |

### RANSACRegressor

**O que faz.** Amostra subconjuntos aleatórios, ajusta o modelo em cada um e fica com o que
tem mais *inliers*. Os outliers são **excluídos** do ajuste final, não apenas atenuados.

**Quando usar.** Contaminação alta e clara (acima de ~20-30% de pontos anômalos), como em
visão computacional ou dados de sensor com falha.

**Evite quando.** Os "outliers" são na verdade parte do fenômeno — você estará descartando
sinal.

### QuantileRegressor

**O que faz.** Modela um **quantil** da distribuição condicional, não a média. Com
`quantile=0.5`, modela a mediana — naturalmente robusta.

**Quando usar.** Quando a mediana é mais representativa que a média, ou quando você quer
construir um **intervalo de predição** ajustando dois modelos (`0.05` e `0.95`).

| Hiperparâmetro | Efeito |
|----------------|--------|
| `quantile` | Qual quantil modelar |
| `alpha` | Regularização L1 |

**Aplicação típica.** Previsão de tempo de entrega com garantia ("90% chegam em até X"),
faixas de preço, planejamento de capacidade.

## GLM (modelos lineares generalizados)

Regressão linear assume erro normal e alvo que pode assumir qualquer valor. Muitos alvos
reais não são assim.

| Modelo | Distribuição do alvo | Aplicação |
|--------|----------------------|-----------|
| `PoissonRegressor` | Contagens: 0, 1, 2, … | Nº de sinistros, visitas, chamados, defeitos |
| `GammaRegressor` | Contínuo positivo, assimétrico, variância proporcional à média² | Valor de sinistro, tempo até evento, custo |
| `TweedieRegressor` | `power` entre 1 e 2: mistura de "zero" com contínuo positivo | Seguros (maioria não usa, quem usa gasta valor contínuo) |

**Por que importa.** Ajustar OLS a uma contagem pode produzir previsões negativas — o que é
impossível por definição. O GLM usa uma função de ligação (log) que garante previsões no
domínio correto e modela a variância adequadamente.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `alpha` | Regularização L2 | `loguniform(1e-4, 1e2)` |
| `power` (Tweedie) | 1 = Poisson, 2 = Gamma, entre = composto | `uniform(1, 1)` |

## Não-linearidade explícita

Modelos lineares só capturam relações lineares — **nas variáveis que você der a eles**. A
saída é construir variáveis não-lineares.

### PolynomialFeatures

Cria potências e interações: de `[a, b]` para `[a, b, a², ab, b²]`.

```python
Pipeline([("prep", prep), ("poly", PolynomialFeatures(degree=2)), ("model", Ridge())])
```

**Cuidado com a explosão combinatória.** Grau 2 sobre 20 colunas gera 230 variáveis; grau 3
gera 1.770. Use `interaction_only=True` para só interações, sem potências, e **sempre com
Ridge** — sem regularização, o overfitting é garantido.

### SplineTransformer

Divide o intervalo de cada variável em segmentos e ajusta polinômios suaves em cada um.
Mais flexível que polinômio global e muito mais estável nas bordas.

**Quando usar.** Relação suave e curva (temperatura × consumo, idade × risco), mantendo
interpretabilidade.

## Métodos de kernel

### SVR

Versão de regressão do SVM. Define um "tubo" de largura `epsilon` em torno da previsão:
erros dentro do tubo não são penalizados.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `C` | Penalidade por sair do tubo | `loguniform(1e-1, 1e4)` |
| `epsilon` | Largura do tubo — maior = modelo mais tolerante e esparso | `loguniform(1e-3, 1e1)` |
| `gamma` | Alcance do kernel RBF | `loguniform(1e-5, 1e1)` |

**Quando usar.** Base pequena e limpa, relação não-linear. Mesma limitação O(n²) do SVC.

### KernelRidge

Ridge no espaço do kernel, com **solução fechada** — mais rápido de treinar que o SVR, mas
sem a esparsidade dos vetores de suporte (usa todos os pontos na predição).

### GaussianProcessRegressor

**O que faz.** Define uma distribuição sobre funções e devolve, além da previsão, a
**incerteza calibrada** em cada ponto — maior onde há menos dados.

**Quando usar.** Poucas centenas de linhas, e a incerteza importa: otimização bayesiana,
experimentos caros, calibração de instrumentos.

**Evite quando.** Mais de ~5 mil linhas — o custo é O(n³) e a memória O(n²).

## Árvores e ensembles em regressão

As mesmas famílias da classificação (DecisionTree, RandomForest, ExtraTrees,
GradientBoosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost) funcionam em regressão
com a mesma lógica de ajuste. Quatro diferenças que valem atenção:

**1. Árvores não extrapolam.** A previsão é sempre a média de uma folha, ou seja, um valor
já visto no treino. Se o alvo tem tendência crescente e você prevê para fora da faixa
observada, o modelo devolve o teto — não a continuação da tendência. Para extrapolar, use
modelo linear.

**2. `max_features` muda de padrão.** Em regressão o padrão do RandomForest é `1.0` (todas
as colunas), não `"sqrt"`.

**3. A função de perda é escolha sua.** `squared_error` é o padrão;
`absolute_error` é robusta a outliers; `poisson` para contagens; `quantile` para
intervalos.

**4. Boosting com perda quantílica dá intervalo de predição:**

```python
q_baixo = GradientBoostingRegressor(loss="quantile", alpha=0.05)
q_alto  = GradientBoostingRegressor(loss="quantile", alpha=0.95)
```

## Transformação do alvo

Alvos como preço, renda, tempo e contagem têm cauda longa à direita. Isso viola a suposição
de erro normal e faz o modelo priorizar os valores altos.

`log1p` comprime a cauda e frequentemente é **o maior ganho isolado do projeto**. O jeito
certo de aplicar:

```python
TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
```

Isso aplica `log1p` no `fit` e `expm1` no `predict` **dentro de cada fold** da validação
cruzada, e devolve as métricas na **escala original** — que é a que o negócio entende.

> [!WARNING]
> Aplicar `np.log1p(y)` manualmente antes do split funciona, mas todas as métricas passam
> a estar em escala log. Um RMSE de 0,21 não diz nada a ninguém.

Como decidir: se `y.skew() > 1`, teste. O notebook 02 tem uma célula que roda o torneio
inteiro com e sem log e mostra o ganho por modelo.

## Métricas de regressão

| Métrica | O que mede | Cuidado |
|---------|------------|---------|
| **RMSE** | Raiz do erro quadrático médio | Dominado por outliers — erro 10× pesa 100× |
| **MAE** | Erro absoluto médio, na unidade do negócio | Não distingue muitos erros pequenos de poucos grandes |
| **MedAE** | Mediana do erro absoluto | Ignora completamente a cauda |
| **MAPE** | Erro percentual médio | **Explode se `y` se aproxima de zero** |
| **R²** | Fração da variância explicada | Não diz se o erro é aceitável; pode ser negativo |
| **RMSLE** | Erro em escala log | Exige `y > 0`; pune subestimativa mais que superestimativa |

**Reporte sempre RMSE + MAE + R² juntos.** Se RMSE >> MAE, existem poucos erros enormes —
sinal de outliers ou de uma faixa do alvo mal modelada.

**Referências que dão sentido ao número:** compare o RMSE com o desvio-padrão do alvo
(razão < 1 significa "melhor que prever a média") e com o RMSE do `DummyRegressor`.

**No sklearn todo scorer é "maior é melhor"** — por isso as métricas de erro aparecem como
`neg_root_mean_squared_error`. Multiplique por −1 para reportar.

### Análise de resíduos

Mais informativa que qualquer métrica isolada. O gráfico de resíduo × previsto revela:

| Padrão | Diagnóstico | Correção |
|--------|-------------|----------|
| Nuvem horizontal sem forma | Modelo bem especificado | — |
| Forma de funil / cone | Heterocedasticidade | `log1p` no alvo |
| Curva em U | Falta não-linearidade | Polinômio, spline ou modelo de árvore |
| Deslocamento sistemático | Viés numa faixa | Investigue por faixa do alvo |

---

# Parte 4 — Clusterização

> Notebook: [`03_clusterizacao.ipynb`](03_clusterizacao.ipynb)

Aprendizado **não supervisionado**: não existe `y`, logo não existe acerto a medir. Quatro
consequências:

1. **Todo `fit` é `fit(X)`** — sem gabarito.
2. **O split muda de função.** Não mede acerto: verifica se a estrutura encontrada **se
   repete** em dados não vistos. Vários algoritmos nem têm `predict`.
3. **A "validação cruzada" é estabilidade.** Reamostra, roda de novo, mede o quanto as
   partições concordam (ARI). Cluster que muda a cada rodada não é cluster, é ruído.
4. **A escala domina o resultado.** Sem `StandardScaler`, a variável de maior amplitude
   define os grupos sozinha.

## Centróide: KMeans e variantes

### KMeans

**O que faz.** Escolhe *k* centros e alterna dois passos até convergir: atribui cada ponto
ao centro mais próximo, depois recalcula cada centro como a média dos seus pontos. Minimiza
a soma das distâncias quadráticas intra-cluster (inércia).

**Quando usar.** Baseline universal de clusterização. Rápido — O(n) —, escala para milhões
de linhas, resultado fácil de explicar (cada cluster tem um "cliente médio").

**Evite quando.** Os grupos não são aproximadamente esféricos e de tamanho parecido — a
suposição embutida no algoritmo. Também é sensível a outliers, que puxam os centróides.

**Pré-processamento.** `"scaled"` — obrigatório.

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `n_clusters` | **Você precisa informar.** Ver [escolha do k](#como-escolher-o-número-de-clusters) | `randint(2, 12)` |
| `n_init` | Execuções independentes; fica com a melhor. Protege de mínimos locais | `[10, 20, 30]` |
| `init` | `k-means++` escolhe centros iniciais distantes entre si — sempre melhor que aleatório | — |

**Aplicação típica.** Segmentação de clientes (RFM), agrupamento de produtos, compressão de
cores, pré-agrupamento antes de modelo supervisionado.

### MiniBatchKMeans

Mesma ideia, mas atualiza os centros usando lotes pequenos em vez do dataset inteiro.
Ordens de magnitude mais rápido, com perda pequena de qualidade. Use acima de ~100 mil
linhas.

### BisectingKMeans

Divide recursivamente: começa com um cluster e vai partindo o maior (ou o de maior inércia)
em dois. Produz uma **hierarquia** e costuma achar clusters mais equilibrados que o KMeans
comum.

## Hierárquico: Agglomerative

**O que faz.** Começa com cada ponto sendo seu próprio cluster e funde os dois mais
próximos, repetidamente, até sobrar um. O histórico de fusões é o **dendrograma** — e você
escolhe onde cortar.

**Quando usar.** Quando a estrutura hierárquica interessa (taxonomias, famílias de
produtos). Quando você quer *ver* a estrutura antes de decidir o número de grupos.

**Evite quando.** Mais de ~10-20 mil linhas — o custo é O(n²) em tempo e memória. E não há
`predict`: rotular dados novos exige um classificador auxiliar.

**O parâmetro que muda tudo é o `linkage`:**

| Linkage | Como mede distância entre clusters | Produz |
|---------|-----------------------------------|--------|
| `ward` | Minimiza o aumento da variância intra-cluster | Clusters compactos e equilibrados (só com métrica euclidiana) |
| `complete` | Distância entre os pontos **mais distantes** | Clusters compactos, sensível a outliers |
| `average` | Distância média entre todos os pares | Meio-termo; aceita métricas não euclidianas |
| `single` | Distância entre os pontos **mais próximos** | Encontra formas alongadas; sofre com efeito de encadeamento |

**Aplicação típica.** Taxonomia de produtos, agrupamento de documentos, análise
filogenética, segmentação exploratória.

## Densidade: DBSCAN, HDBSCAN, OPTICS

Definem cluster como **região densa** separada por regiões esparsas. Três vantagens sobre o
KMeans: não exigem *k*, encontram formatos arbitrários e **identificam ruído**
explicitamente (rótulo `-1`).

### DBSCAN

**O que faz.** Um ponto é "central" se tem pelo menos `min_samples` vizinhos dentro do raio
`eps`. Pontos centrais conectados formam um cluster; o que sobra é ruído.

**Quando usar.** Formatos irregulares (o exemplo clássico são dois anéis concêntricos, que
o KMeans nunca separa). Quando identificar outliers faz parte do objetivo.

**Evite quando.** Os clusters têm **densidades muito diferentes** — um único `eps` não
serve para todos. Nesse caso, use HDBSCAN ou OPTICS.

| Hiperparâmetro | Efeito | Como escolher |
|----------------|--------|---------------|
| `eps` | Raio da vizinhança. **O parâmetro crítico** | Curva k-distância: plote a distância ao k-ésimo vizinho ordenada; o joelho é o `eps` |
| `min_samples` | Densidade mínima | Comece com `2 × n_dimensões` |

O notebook 03 traz a curva k-distância pronta na célula 4.4.

**Aplicação típica.** Detecção de anomalias, agrupamento espacial (GPS, sismologia),
identificação de regiões de interesse em imagens.

### HDBSCAN

**O que faz.** Versão hierárquica do DBSCAN: constrói a hierarquia sobre densidades
variáveis e extrai os clusters mais estáveis. **Dispensa o `eps`** — o parâmetro mais
difícil de acertar.

**Quando usar.** Praticamente sempre que consideraria DBSCAN. Lida com densidades diferentes,
tem menos parâmetros e o principal deles (`min_cluster_size`) é intuitivo: "qual o menor
grupo que me interessa?".

| Hiperparâmetro | Efeito | Faixa |
|----------------|--------|-------|
| `min_cluster_size` | Menor grupo aceitável | `randint(5, 100)` |
| `min_samples` | Rigor na definição de ruído. Maior = mais pontos viram ruído | `[None, 5, 10, 25]` |
| `cluster_selection_method` | `eom` = grupos amplos; `leaf` = grupos finos | — |

Disponível nativamente no scikit-learn desde a versão 1.3.

### OPTICS

Ordena os pontos por acessibilidade e produz um *reachability plot* — de onde se extraem
clusters em múltiplas escalas de densidade. Mais informativo, porém mais lento e com mais
parâmetros que o HDBSCAN.

## Probabilístico: Gaussian Mixture

### GaussianMixture

**O que faz.** Assume que os dados vêm de uma mistura de *k* distribuições gaussianas e
estima seus parâmetros por Expectation-Maximization. Diferente do KMeans, a atribuição é
**suave**: cada ponto tem uma probabilidade de pertencer a cada cluster.

**Quando usar.** Clusters elípticos, de tamanhos e orientações diferentes — o KMeans só
enxerga esferas. Quando a probabilidade de pertencimento importa (um cliente 60%/40% entre
dois segmentos é informação útil). E quando você quer um critério **principiado** para
escolher *k*: BIC e AIC.

**Evite quando.** Os dados claramente não são gaussianos, ou há muito mais dimensões que
amostras.

| Hiperparâmetro | Efeito |
|----------------|--------|
| `n_components` | Número de gaussianas |
| `covariance_type` | `full` = elipse livre; `tied` = mesma forma para todas; `diag` = eixos alinhados; `spherical` = círculo (equivale ao KMeans) |
| `n_init` | Execuções independentes — EM tem mínimos locais |

```python
gmm.bic(X)   # menor é melhor — o critério mais defensável para escolher k
gmm.predict_proba(X)   # probabilidade por cluster
```

**Aplicação típica.** Segmentação com transição gradual entre grupos, modelagem de
subpopulações, detecção de anomalia por baixa verossimilhança.

### BayesianGaussianMixture

Versão bayesiana que **poda componentes desnecessários** sozinha: você informa um teto para
`n_components` e ela zera o peso dos que não são usados. Útil quando você não faz ideia do
número de grupos.

## Grafo: Spectral Clustering

**O que faz.** Constrói um grafo de similaridade entre os pontos, calcula os autovetores do
Laplaciano desse grafo e roda KMeans nesse novo espaço. A projeção espectral "desenrola"
estruturas que são inseparáveis no espaço original.

**Quando usar.** Formatos entrelaçados — anéis concêntricos, meias-luas, espirais. É o
algoritmo que resolve os casos onde todos os outros falham visivelmente.

**Evite quando.** Mais de ~5-10 mil linhas: exige decomposição de uma matriz n×n, custo
O(n³).

| Hiperparâmetro | Efeito |
|----------------|--------|
| `n_clusters` | Obrigatório |
| `affinity` | `nearest_neighbors` (esparso, mais estável) ou `rbf` (denso) |
| `n_neighbors` | Vizinhos no grafo — controla o quanto a estrutura local pesa |

**Aplicação típica.** Segmentação de imagem, comunidades em redes sociais, agrupamento de
trajetórias.

## Outras famílias

### MeanShift

Busca as **modas** da densidade deslocando iterativamente cada ponto na direção de maior
concentração. Descobre o número de clusters sozinho; o resultado depende inteiramente do
`bandwidth` (use `estimate_bandwidth` como ponto de partida). Custo alto.

### Birch

Constrói uma árvore de resumos (CF-tree) em **uma única passada** pelos dados. Pensado para
volumes que não cabem na memória. Costuma ser usado como pré-agrupador, seguido de KMeans
sobre os subclusters.

### AffinityPropagation

Os pontos "trocam mensagens" até elegerem **exemplares reais** como centros — diferente do
KMeans, cujo centróide é um ponto médio artificial. Não exige *k*, mas é O(n²) em memória e
tende a produzir clusters demais.

## Como escolher o número de clusters

Nenhum critério isolado basta. Use os quatro e procure convergência:

**1. Curva do cotovelo (inércia).** Plote a inércia contra *k*. A inércia sempre cai — o
que se procura é o "cotovelo", o ponto onde a queda desacelera. Subjetivo, mas útil.

**2. Silhueta.** Mede o quanto cada ponto está mais próximo do seu cluster que do vizinho
mais próximo. Varia de −1 a 1; maior é melhor. **Viés conhecido:** favorece clusters
convexos e esféricos, então tende a preferir *k* pequeno.

**3. BIC / AIC (só GMM).** Penaliza a complexidade do modelo. É o critério mais
principiado da lista — e o único que tem fundamento estatístico real.

**4. Estabilidade.** Rode em duas subamostras sobrepostas e meça o ARI entre as partições
na interseção. **Este é o critério mais confiável na prática:**

| ARI médio | Leitura |
|-----------|---------|
| > 0,75 | Estrutura sólida |
| 0,50 – 0,75 | Estrutura moderada — cuidado ao usar em decisão |
| < 0,50 | O algoritmo está inventando os grupos |

**5. Interpretabilidade.** O critério final. Se *k*=7 tem silhueta ótima mas produz três
grupos que ninguém sabe descrever, e *k*=4 gera perfis que o time de negócio reconhece na
hora, use 4. Cluster que não vira ação não vale nada.

## Métricas de clusterização

**Internas** — só usam `X` e os rótulos previstos. O caso normal.

| Métrica | Faixa | Direção | Viés |
|---------|-------|---------|------|
| Silhueta | [−1, 1] | maior | Favorece formas convexas |
| Calinski-Harabasz | [0, ∞) | maior | Favorece *k* pequeno |
| Davies-Bouldin | [0, ∞) | **menor** | Também favorece convexidade |
| Inércia (SSE) | [0, ∞) | menor | Cai sempre com *k* — só serve para o cotovelo |
| BIC / AIC | ℝ | menor | Só para mistura gaussiana |

**Externas** — exigem um rótulo verdadeiro. Só quando existe gabarito.

| Métrica | O que mede |
|---------|------------|
| **ARI** (Adjusted Rand Index) | Concordância entre partições, corrigida pelo acaso. 1 = idênticas, 0 = acaso |
| **NMI / AMI** | Informação mútua normalizada entre as partições |
| Homogeneidade | Cada cluster contém uma única classe |
| Completude | Cada classe cai num único cluster |
| V-measure | Média harmônica das duas anteriores |

> [!NOTE]
> Todas essas métricas são **invariantes à numeração** dos clusters. É por isso que se usa
> ARI, e não acurácia: o cluster 0 não tem obrigação nenhuma de corresponder à classe 0.

---

# Parte 5 — Otimização de hiperparâmetros

| Método | Como funciona | Use quando |
|--------|---------------|------------|
| `GridSearchCV` | Testa **todas** as combinações | Poucos parâmetros (≤ 3) com poucos valores cada |
| `RandomizedSearchCV` | Amostra `n_iter` combinações do espaço | **Padrão.** Espaço grande, aceita distribuições contínuas |
| `HalvingRandomSearchCV` | Avalia muitos candidatos com poucos dados e elimina os piores | Muitos candidatos, dataset grande. 3-10× mais rápido |
| Optuna (bayesiana) | Usa os resultados anteriores para escolher onde testar | Cada `fit` é caro; vale investir em escolher bem |

**Por que Random costuma vencer Grid.** Se apenas 2 dos 5 hiperparâmetros realmente
importam, o Grid desperdiça a maior parte das avaliações variando parâmetros irrelevantes.
O Random cobre mais valores distintos dos parâmetros que importam, com o mesmo orçamento. E
só o Random aceita distribuição contínua:

```python
"model__C": loguniform(1e-4, 1e3)      # amostra em escala log — o certo para C, alpha, gamma
"model__max_depth": randint(2, 12)      # inteiros
"model__subsample": uniform(0.5, 0.5)   # contínuo em [0.5, 1.0] — note: (início, LARGURA)
```

> [!TIP]
> `uniform(a, b)` no scipy é `[a, a+b]`, não `[a, b]`. `uniform(0.5, 0.5)` cobre
> [0.5, 1.0]. É a confusão mais comum ao escrever grades.

**Estratégia recomendada:**

1. `RandomizedSearchCV` amplo (`n_iter=60`, faixas largas) para encontrar a região boa;
2. `GridSearchCV` estreito em volta do vencedor para refinar — opcional, ganho pequeno.

**O que otimizar junto.** A grade pode incluir o pré-processamento, e frequentemente o
ganho está ali:

```python
{**GRIDS["RandomForest"], "prep__num__scaler": [StandardScaler(), RobustScaler()]}
```

**Escala log vs. linear.** Parâmetros de regularização (`C`, `alpha`, `gamma`,
`learning_rate`) devem ser amostrados em escala **logarítmica**: a diferença entre 0,001 e
0,01 importa tanto quanto entre 1 e 10. `loguniform` faz isso; `uniform` desperdiçaria
quase todas as amostras na faixa alta.

---

# Parte 6 — Guia rápido de decisão

## Por onde começar

| Situação | Comece por |
|----------|------------|
| Tabular, qualquer tamanho | **HistGradientBoosting / LightGBM** |
| Precisa explicar a decisão | `LogisticRegression`, `Ridge`, `DecisionTree` raso |
| Poucas linhas (< 1.000) | `LogisticRegression`, `Ridge`, `SVC`, `LDA`, `GaussianNB` |
| Muitas colunas, poucas linhas | `LogisticRegression(penalty="l1")`, `Lasso`, `LinearSVC` |
| Muitas linhas (> 1M) | `SGDClassifier`, `LightGBM`, `MiniBatchKMeans` |
| Texto (TF-IDF) | `LinearSVC`, `ComplementNB`, `LogisticRegression` |
| Categóricas de alta cardinalidade | `CatBoost`, `HistGradientBoosting` |
| Precisa extrapolar | Modelo linear — árvores não extrapolam |
| Precisa de intervalo de predição | Regressão quantílica, `BayesianRidge`, `GaussianProcess` |
| Alvo = contagem | `PoissonRegressor`, `HistGB(loss="poisson")` |
| Outliers pesados | `Huber`, `RANSAC`, `loss="absolute_error"`, HDBSCAN |
| Não sabe quantos grupos | `HDBSCAN`, `BayesianGMM` |
| Grupos de formato irregular | `DBSCAN`, `HDBSCAN`, `SpectralClustering` |

## Sintomas e correções

| Sintoma | Causa provável | Correção |
|---------|----------------|----------|
| CV ótima, teste ruim | Vazamento, ou busca superajustada | Pré-processo dentro do Pipeline; reduza `n_iter`; CV repetida |
| Treino >> validação | Overfitting | Mais regularização, menos profundidade, mais dados |
| Treino ≈ validação, ambos ruins | Underfitting | Mais capacidade, mais variáveis, interações |
| Acurácia alta, recall zero | Desbalanceamento | `class_weight`, troque a métrica, ajuste o threshold |
| Métrica instável entre folds | Dataset pequeno | `RepeatedKFold`, mais folds |
| Ótimo em CV, péssimo em produção | *Drift*, ou grupo vazando entre os folds | Split por tempo ou por grupo |
| R² negativo | Pior que prever a média | Verifique pré-processo e vazamento invertido |
| Resíduo em funil | Heterocedasticidade | `log1p` no alvo |
| Resíduo em U | Falta não-linearidade | Polinômio, spline, ou modelo de árvore |
| Um cluster com 95% dos pontos | Escala ou outlier dominando | `StandardScaler`; trate outliers |
| DBSCAN devolve tudo como ruído | `eps` pequeno demais | Curva k-distância |
| Clusters mudam a cada execução | Estrutura frágil, ou `n_init` baixo | Suba `n_init`; teste estabilidade; revise o *k* |

## Os erros que invalidam um projeto

1. **Pré-processar antes do split.** Escala, imputação e encoding ajustados no dataset
   inteiro contaminam a validação.
2. **SMOTE antes da validação cruzada.** Exemplos sintéticos vazam para os folds de teste.
3. **Escolher o threshold olhando o teste.** Calibre na validação; o teste é para reportar.
4. **Embaralhar série temporal.** O modelo aprende o futuro.
5. **Deixar a mesma entidade nos dois lados do split.** O modelo decora, não generaliza.
6. **Abrir o teste mais de uma vez.** Olhar, ajustar e olhar de novo transforma o teste em
   validação — e a estimativa deixa de ser honesta.
7. **Deixar um ID numérico entre as variáveis.** Especialmente destrutivo em clusterização.

---

## Ambiente validado

Testado com **48 verificações** cobrindo as APIs que os notebooks usam.

| Componente | Versão |
|------------|--------|
| Python | 3.14.6 |
| scikit-learn | 1.9.0 |
| numpy · pandas · scipy | 2.4.6 · 3.0.5 · 1.18.0 |
| matplotlib | 3.11.1 |
| xgboost · lightgbm · catboost | 3.4.0 · 4.7.0 · 1.2.10 |
| imbalanced-learn | 0.14.2 |

> [!NOTE]
> - **pandas 3.0** mudou comportamentos (Copy-on-Write por padrão, novo dtype de string).
>   Tudo que os notebooks usam foi verificado nessa versão.
> - **O truque `cv=[(slice(None), slice(None))]`**, comum em receitas de clusterização na
>   internet, **parou de funcionar com DataFrame** a partir do scikit-learn 1.7. O notebook
>   03 usa `[(np.arange(n), np.arange(n))]`, com a nota explicando o porquê.

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
└── README.md                 # este guia
```
