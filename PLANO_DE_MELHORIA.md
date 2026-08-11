# Plano de melhoria — Master Script scikit-learn

Análise técnica dos três notebooks, do `README.md`, do `INSTALACAO.md` e da configuração
do projeto, com um plano de correção priorizado.

**Data da análise:** 2026-08-11 · **Commit analisado:** `7980fcb`

---

## Sumário executivo

O conteúdo é forte. A estrutura de seis etapas repetida nos três notebooks, a separação em
famílias matemáticas, a insistência no Pipeline como barreira de vazamento e o cuidado com
as armadilhas (SMOTE antes da CV, threshold no teste, silhueta em espaço projetado) colocam
este material acima da média do que circula sobre scikit-learn em português.

O problema não é o conteúdo — é que **o código nunca foi executado de ponta a ponta desde a
última revisão**, e há três defeitos que impedem isso. O notebook 02 quebra na célula 4.2 e
não produz nada das etapas 4, 5 e 6. Os notebooks 01 e 02 travam indefinidamente no
LightGBM. E o dataset fictício do notebook 01 não tem sinal nenhum: nenhuma das 20 famílias
supera o acaso, e o `DummyClassifier` termina em 3º lugar no próprio torneio.

Nada disso é difícil de corrigir. A Fase 0 abaixo leva cerca de 30 minutos e resolve os três.
O que falta de verdade é o mecanismo que teria pego os dois primeiros: um teste que executa
os notebooks.

| Eixo | Estado |
|---|---|
| Conteúdo conceitual (README) | Muito bom — correções pontuais |
| Cobertura de famílias e grades | Muito boa — 61 pipelines, faltam 4 grades |
| Código executável | **Quebrado** — 1 blocador, 1 travamento, 1 risco de `KeyError` |
| Dados de exemplo | Fracos no nb 01 (sem sinal), limítrofes no nb 02 |
| Documentação | Boa, com 3 afirmações desatualizadas ou não verificáveis |
| Infraestrutura (testes, CI, higiene) | **Ausente** |

### Como estes achados foram verificados

Não são leitura de código. Foi montado um ambiente com `scikit-learn 1.9.0`, `pandas 3.0.5`,
`numpy 2.5.2`, `xgboost 3.4.0`, `lightgbm 4.7.0`, `catboost 1.2.10` e `imbalanced-learn`, e
as células dos três notebooks foram executadas em sequência num mesmo namespace. Cada
defeito abaixo traz a evidência da execução.

---

## Fase 0 — Defeitos que impedem a execução

### 0.1 · Notebook 02: nome de scorer inválido derruba as etapas 4, 5 e 6

**Onde:** `02_regressao.ipynb`, célula 4.1 (`SCORING`).

```python
SCORING = {
    ...
    "max_error": "max_error",     # <- não existe
}
```

`max_error` não está em `sklearn.metrics.get_scorer_names()`. O nome correto é
**`neg_max_error`**. Como todo scorer no sklearn é "maior é melhor", a versão registrada é a
negativa — exatamente a convenção que o próprio notebook explica duas células antes.

**Efeito medido.** `cross_validate` levanta `ValueError` na validação do scoring, *antes* de
qualquer `fit`. O `try/except` de `rodar_torneio` engole o erro e imprime `FALHOU` para as
27 famílias. `ranking` sai vazio e a célula 4.2 morre com `KeyError: 'rmse_mean'`. A partir
daí, cascata:

```
CELL 20 FAIL  KeyError: 'rmse_mean'          <- 4.2 torneio
CELL 21 FAIL  KeyError: 'rmse_mean'          <- 4.3 comparação alvo bruto x log
CELL 22 FAIL  KeyError: 'modelo'             <- 4.4 escolha da vencedora
CELL 23 FAIL  NameError: MELHOR_FAMILIA      <- 4.5 curva de aprendizado
CELL 27 FAIL  NameError: MELHOR_FAMILIA      <- 5.3 a busca
CELL 28 FAIL  NameError: search              <- 5.4 inspeção
CELL 31..37   NameError: best_model / y_pred <- toda a etapa 6
```

**Treze células.** Metade do notebook não roda.

**Correção:**

```python
"max_error": "neg_max_error",
```

Com isso o `sinal = -1 if SCORING[m].startswith("neg_")` de `rodar_torneio` passa a tratar a
métrica corretamente — hoje, mesmo que o nome fosse aceito, ela seria reportada negativa.

**Verificação:** com esse único caractere corrigido, o notebook 02 percorreu o torneio
inteiro (27 famílias, `RMSE` entre 61k e 225k, `R²` de −3,86 a 0,60).

---

### 0.2 · `N_JOBS = -1` no estimador *e* na validação cruzada trava o LightGBM

**Onde:** os três notebooks — `N_JOBS = -1` na célula 0, repassado tanto ao estimador
(`LGBMClassifier(n_jobs=N_JOBS)`) quanto ao `cross_validate(n_jobs=N_JOBS)` e ao
`RandomizedSearchCV(n_jobs=N_JOBS)`.

Cada worker do joblib abre um pool OpenMP do tamanho da máquina. Em `k` núcleos isso gera
`k × k` threads disputando `k` núcleos. Com LightGBM o resultado não é lentidão — é
travamento efetivo.

**Medição** (4 núcleos, 800 linhas, `LGBMClassifier(n_estimators=600)`, CV de 5 folds):

| `n_jobs` do modelo | `n_jobs` da CV | Tempo |
|---|---|---|
| 1 | 1 | 1,7 s |
| −1 | 1 | 1,3 s |
| 1 | −1 | 2,0 s |
| **−1** | **−1** | **não terminou em 3,5 min** (4 processos a 99% de CPU) |

A última linha é a configuração dos notebooks. Nas execuções completas, tanto o nb 01 quanto
o nb 02 pararam exatamente na linha do LightGBM do torneio e nunca saíram dela. O nb 03
apresenta o mesmo padrão no `OPTICS`, agravado porque a função `estabilidade` refaz o ajuste
5 vezes por família.

**Correção.** Paralelizar num nível só. O externo rende mais, porque cobre todos os
estimadores:

```python
# célula 0 — Config global
N_JOBS       = -1   # usado APENAS por cross_validate / RandomizedSearchCV
N_JOBS_MODEL = 1    # dentro do estimador: 1 evita sobreposição com o paralelismo da CV
```

e trocar `n_jobs=N_JOBS` por `n_jobs=N_JOBS_MODEL` em `KNeighbors*`, `RandomForest*`,
`ExtraTrees*`, `LGBM*`, `XGB*`, `DBSCAN`, `HDBSCAN`, `OPTICS`, `MeanShift`, `Spectral`.
Vale um parágrafo no README: é um erro que quase todo mundo comete e quase ninguém
diagnostica.

---

### 0.3 · `GRIDS[MELHOR_FAMILIA]` levanta `KeyError` quando a vencedora não tem grade

**Onde:** nb 01 célula 5.3, nb 02 célula 5.3, nb 03 célula 5.3.

Famílias declaradas em `PIPELINES` mas ausentes de `GRIDS`:

| Notebook | Sem grade |
|---|---|
| 01 | `Dummy` |
| 02 | `Dummy`, `LinearRegression` |
| 03 | `Agglomerative_average` |

`Dummy` e `LinearRegression` de fato não têm hiperparâmetro que valha buscar — o problema é
que a etapa 5 não trata esse caso. E **não é hipotético**: no torneio do próprio notebook 01,
com os dados fictícios atuais, o `Dummy` termina em 3º lugar (ver 2.1). Com outra semente,
ele lidera e a célula 5.3 quebra.

**Correção:**

```python
GRIDS_VAZIAS = {"Dummy", "LinearRegression"}   # nada a otimizar

if MELHOR_FAMILIA in GRIDS_VAZIAS or MELHOR_FAMILIA not in GRIDS:
    print(f"{MELHOR_FAMILIA} não tem grade de busca. "
          f"Se um baseline venceu o torneio, o problema está nos dados ou nas variáveis, "
          f"não nos hiperparâmetros.")
    grid_vencedor = {}
else:
    grid_vencedor = GRIDS[MELHOR_FAMILIA]
```

No nb 03, dar a `Agglomerative_average` a mesma grade de `Agglomerative_ward` (com
`linkage` restrito aos que aceitam `metric` livre) resolve.

**Critério de aceite da Fase 0:** os três notebooks executam de ponta a ponta, sem exceção,
em menos de 15 minutos cada.

---

## Fase 1 — O teste que teria pego tudo isso

Nenhum dos defeitos da Fase 0 é sutil. Todos apareceriam na primeira execução automatizada.
Esta é a mudança de maior retorno do plano inteiro.

### 1.1 · `tests/test_notebooks.py`

Quatro verificações, todas baratas, que não exigem executar o notebook completo:

```python
# a) todo pipeline declarado tem grade (ou está na lista de isentos)
# b) todo nome de scorer usado existe em sklearn.metrics.get_scorer_names()
# c) 3 amostras de cada grade passam por clone(pipe).set_params(**params)
# d) toda chave de grade existe em pipe.get_params(deep=True)
```

O item (b) teria pego o defeito 0.1 em menos de um segundo. O item (a) teria pego o 0.3.
O item (d) é a rede contra erros de digitação em `model__` e `prep__num__`.

A extração é direta: `json.load` do `.ipynb`, executar as células até a que define `GRIDS`,
pular as células de torneio e busca.

### 1.2 · Execução completa em modo rápido

Introduzir uma variável de ambiente lida na célula 0 dos três notebooks:

```python
import os
MODO_RAPIDO = os.getenv("MASTER_SCRIPT_FAST") == "1"

N_ITER_BUSCA   = 5   if MODO_RAPIDO else 60
N_LINHAS       = 300 if MODO_RAPIDO else 1_000
N_RODADAS_ESTAB = 2  if MODO_RAPIDO else 10
```

Com isso, `jupyter nbconvert --execute --to notebook` roda os três em CI em poucos minutos.
Sem isso, CI é inviável: o torneio do nb 01 encadeia 20 famílias, incluindo CatBoost com 600
iterações em 5 folds, e o nb 03 refaz cada ajuste 5 vezes para medir estabilidade.

### 1.3 · GitHub Actions

```yaml
- uses: astral-sh/setup-uv@v5
- run: uv sync --all-extras
- run: uv run pytest tests/
- run: MASTER_SCRIPT_FAST=1 uv run jupyter nbconvert --execute --to notebook \
         --output /tmp/out.ipynb 0*.ipynb
```

### 1.4 · `nbstripout` via pre-commit

O `01_classificacao.ipynb` carrega saídas de uma execução abortada em 7 células —
incluindo um **traceback de `KeyboardInterrupt`** na célula 4.2 e o caminho local
`~/Documentos/skitlearn_mastre_script/.venv/lib/python3.14/...`. Os notebooks 02 e 03 estão
limpos. O estado é inconsistente e polui todo diff futuro.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.8.1
    hooks: [{id: nbstripout}]
```

**Critério de aceite:** um PR que reintroduza `"max_error"` falha no CI.

---

## Fase 2 — Dados de exemplo que demonstram o que o texto ensina

### 2.1 · O alvo do notebook 01 é ruído puro

```python
"churn": rng.choice([0, 1], n, p=[.85, .15]),   # independente de TODAS as features
```

O `churn` é sorteado sem nenhuma relação com `idade`, `renda`, `score`, `tempo_casa`,
`n_produtos`, `regiao`, `canal` ou `plano`. Não há o que aprender. Resultado do torneio na
execução completa:

| Posição | Modelo | ROC-AUC |
|---|---|---|
| 1º | SVC_rbf | 0,5586 |
| 2º | KNN | 0,5293 |
| **3º** | **Dummy** | **0,5273** |
| 4º | XGBoost | 0,5194 |
| … | (as 20 famílias) | 0,43 – 0,56 |

Isso não é um detalhe estético. Tudo que vem depois — a busca de hiperparâmetros, a matriz
de confusão, o ajuste de threshold por custo, a importância por permutação, a análise dos
"erros mais graves" — está ensinando a interpretar as saídas de um modelo que não aprendeu
nada. A curva de calibração de um classificador aleatório e o gráfico de importância de
variáveis irrelevantes ensinam a ler ruído como se fosse resultado.

**Correção** — gerar o alvo a partir de um logito das features, mantendo os 15% de
prevalência e um sinal moderado (ROC-AUC realista, na faixa de 0,75–0,85):

```python
logito = (-1.9
          + 0.020 * (df["score"].fillna(600) - 600) / 90 * -1.4   # score baixo -> mais churn
          + 0.9  * (df["tempo_casa"] < 1.5)
          - 0.5  * (df["n_produtos"] >= 3)
          + 0.6  * (df["canal"] == "loja")
          + rng.normal(0, 0.7, n))
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int)
```

Ajustar o intercepto até `df["churn"].mean()` ficar perto de 0,15. O ganho pedagógico é
grande: a matriz de confusão passa a ter estrutura, o threshold por custo passa a deslocar
de fato o ponto de operação, e a importância por permutação aponta as variáveis plantadas —
o leitor consegue conferir se entendeu.

### 2.2 · O alvo do notebook 02 não atinge o limiar que o próprio notebook define

O notebook imprime `skew > 1 -> forte candidato a log1p` e o alvo gerado tem **skew =
0,757**. A seção 4.3, que compara o torneio com e sem `log1p` no alvo, fica sem demonstrar
ganho — que é justamente o ponto que o cabeçalho chama de "o ganho isolado mais alto do
projeto".

**Correção:** subir o multiplicador lognormal do ruído de `rng.lognormal(0, .18, n)` para
`rng.lognormal(0, .45, n)`, ou aplicar o ruído a uma base já exponencial. Meta: skew entre
1,5 e 2,5.

### 2.3 · Custo de execução

O torneio do nb 01 encadeia CatBoost (600 iterações), SVC com `probability=True` e MLP em 5
folds. O nb 03 chama `estabilidade(n_rodadas=5)` para 14 famílias — 10 ajustes extras cada,
incluindo `OPTICS`, `Spectral` e `AffinityPropagation`, todos O(n²). Resolvido pela flag
`MODO_RAPIDO` da Fase 1.2, mas vale um aviso explícito no cabeçalho de cada notebook:
"a célula 4.2 leva de X a Y minutos".

---

## Fase 3 — Correções de método e robustez

Nenhuma destas quebra a execução. Todas afetam a correção do que é ensinado.

### 3.1 · `warnings.filterwarnings("ignore")` global

Aparece na célula 4.2 dos três notebooks e nunca é revertido. Some com exatamente os avisos
que interessam: `ConvergenceWarning` do `saga` com L1, do `Lasso` com `alpha` baixo, do
`MLP` com `lbfgs` e `early_stopping`. São o sinal de que a grade está mal calibrada.

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    ...   # torneio aqui dentro; ConvergenceWarning continua visível
```

### 3.2 · `categorical_features` por posição em vez de nome

```python
HistGradientBoostingClassifier(
    categorical_features=[len(NUM_FEATURES) + i for i in range(len(CAT_FEATURES))],
)
```

Funciona hoje — verificado no sklearn 1.9. Mas depende de o `ColumnTransformer` manter a
ordem `num` antes de `cat`, e quebra em silêncio (tratando numérica como categórica) se
alguém reordenar as listas. Como o `prep` já tem `set_output(transform="pandas")`, os nomes
chegam intactos:

```python
categorical_features=CAT_FEATURES,   # verificado: funciona com o prep "native"
```

### 3.3 · `scale_pos_weight` congelado na definição do dicionário

```python
"XGBoost": Pipeline([..., XGBClassifier(
    scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()), ...)]),
```

Calculado quando a célula 3.3 roda, o que amarra o dicionário `PIPELINES` à existência de
`y_train` e ao momento em que a célula 2.1 executou. Além disso, a grade da etapa 5
sobrescreve com `[1, 3, 5, 10]`, deixando o cálculo inútil. Ou usar uma constante calculada
uma vez com nome próprio (`PESO_POS`), ou remover o parâmetro e deixar só na grade.

### 3.4 · Threshold ajustado no conjunto de teste (nb 01, 6.5)

O notebook faz a coisa certa ao avisar, em comentário, que isso é vazamento leve. Mas o
código que ele oferece para copiar é o vazado. Vale inverter: entregar a versão correta como
padrão e deixar a do teste como variante.

```python
from sklearn.model_selection import cross_val_predict
proba_cv = cross_val_predict(best_model, X_train, y_train, cv=CV,
                             method="predict_proba", n_jobs=N_JOBS)[:, 1]
# escolher THRESHOLD sobre (y_train, proba_cv); usar o teste só para reportar
```

### 3.5 · Ajustes menores verificados

| Onde | Problema | Correção |
|---|---|---|
| nb 02, 6.1 | `R2_ajustado` usa `X_test.shape[1]` (colunas brutas), não o nº de features pós-encoding | `best_model[:-1].transform(X_test).shape[1]` |
| nb 02, 6.4 | `lim_inf` / `lim_sup` sobrescrevem as variáveis de outlier definidas em 1.3 | renomear para `q05` / `q95` |
| nb 03, 6.6 | `clone(MELHOR_PIPE).fit(X)` seguido de `.fit_predict(X)` — dois ajustes completos, e para algoritmos sem `predict` os rótulos salvos podem não corresponder ao modelo persistido | ajustar uma vez, guardar `labels_` |
| nb 03, 4.1 | `metricas_internas` não devolve `menor_cluster` no ramo degenerado | incluir a chave com `np.nan` |
| nb 03, 6.1/6.6 | `prep_ref` e `prep_full` reajustam o pré-processador fora do pipeline; a silhueta do artefato usa um `prep` ajustado no treino aplicado a `X` inteiro | reaproveitar `modelo_producao[:-1]` |
| nb 01/02, célula 0 | `np.random.seed(RANDOM_STATE)` (API legada) convive com `default_rng` | manter só o `Generator` |
| nb 01, 5.1 | `AdaBoost` → `[DecisionTreeClassifier(max_depth=d) for d in ...]` sem `random_state` | passar `random_state=RANDOM_STATE` |

---

## Fase 4 — Documentação

### 4.1 · Números que não conferem

| Afirmação | Onde | Real |
|---|---|---|
| nb 02 tem "25 famílias" | README (2 lugares) | **27** pipelines em `PIPELINES` |
| nb 03 tem "15 famílias" | README (2 lugares) | **14** pipelines em `PIPELINES` |
| nb 01 tem "20 famílias" | README | 20 ✓ |

A contagem deve virar assertiva de teste (Fase 1.1), não número escrito à mão em quatro
lugares.

### 4.2 · "Testado com 48 verificações" sem artefato

O README afirma, na seção *Ambiente validado*: **"Testado com 48 verificações cobrindo as
APIs que os notebooks usam."** Não existe nenhum arquivo de teste no repositório. Sejam
quais forem essas 48 verificações, elas não são reproduzíveis nem executáveis por quem clona
o projeto — e o `max_error` do defeito 0.1 passou por elas.

Duas saídas honestas: publicar o script como `tests/` (o que a Fase 1 faz de qualquer forma,
e aí o número vira a contagem real do `pytest`), ou remover a frase.

### 4.3 · `INSTALACAO.md` fala da máquina do autor

Três trechos presos a um ambiente específico, que confundem quem clona:

- "Já está instalado nesta máquina (`uv 0.11.25`)"
- "**Estado atual desta máquina:** o ambiente foi sincronizado com `--all-extras`…"
- "O ambiente ocupa … cerca de **1,6 GB** com todos os extras" (ok como estimativa, desde
  que apresentada como tal)

Reescrever na segunda pessoa, como instrução, não como relato.

### 4.4 · Versões fixadas em três lugares

`3.14.6` / `1.9.0` / `2.4.6` aparecem nos badges do README, na tabela *Ambiente validado* do
README e na tabela do `INSTALACAO.md`. Três pontos para desatualizar em conjunto. Manter uma
fonte (a tabela do README) e fazer as outras apontarem para ela — ou gerar a tabela a partir
do `uv.lock` num passo de CI.

> Nota menor: `INSTALACAO.md` diz que a tabela reflete o `uv.lock`, e o lock traz
> `numpy 2.4.6`; o resolvedor num Python 3.12 entrega `2.5.2`. O teto `<2.5` do
> `pyproject.toml` só vale enquanto o `numba` exigir isso — vale um comentário datado.

### 4.5 · Duplicação README ↔ apêndices dos notebooks

As tabelas "Qual família tentar primeiro" e "Sintomas e correções" existem em quatro
versões: uma no README (Parte 6) e uma no apêndice de cada notebook, com pequenas
divergências entre si. Escolher uma fonte de verdade — sugestão: manter no README, e nos
apêndices deixar só o que é específico daquele notebook (multiclasse, séries temporais,
erros que invalidam a clusterização).

### 4.6 · Faltando

- **`LICENSE`** — material de estudo público sem licença é material que ninguém pode
  reutilizar com segurança. MIT ou CC-BY-4.0.
- Seção **"como manter"**: como rodar os testes, como atualizar as versões, o que fazer
  antes de commitar (o hook do `nbstripout`).
- Os 1.458 linhas do README num arquivo só funcionam como referência de busca (`Ctrl+F`),
  mas mal como leitura. Opcional: dividir em `docs/01-fundamentos.md`, `docs/02-…` com o
  índice no README. Só depois de tudo acima — é a mudança de menor retorno da lista.

---

## Fase 5 — Opcional: código compartilhado

`build_preprocessor` aparece três vezes com variações, e `metricas_internas`, `estabilidade`
e os scorers de clusterização vivem só dentro do nb 03. Um pacote pequeno —
`sklearn_master/preprocessing.py`, `metrics.py`, `grids.py` — permitiria testar essas funções
diretamente, em vez de por execução de notebook.

**Contra-argumento, que vale considerar:** o valor de um guia de consulta está em cada
notebook ser autossuficiente para copiar e colar. Extrair para um pacote transforma
"copio a célula" em "instalo o pacote", o que contraria o propósito declarado no cabeçalho.

Meio-termo recomendado: manter o código duplicado nos notebooks (é o produto) e extrair
para `tests/` apenas as *grades* (`GRIDS`), que são dados e não código, para poderem ser
validadas sem executar nada.

---

## Ordem de execução sugerida

| Fase | Escopo | Esforço | Por que nesta ordem |
|---|---|---|---|
| **0** | 3 defeitos que impedem a execução | ~30 min | Sem isso, metade do nb 02 não roda e dois notebooks travam |
| **1** | Testes + `MODO_RAPIDO` + CI + `nbstripout` | 3–4 h | Trava a regressão antes de mexer em qualquer outra coisa |
| **2** | Dados sintéticos com sinal (nb 01 e 02) | 2 h | Maior ganho didático por hora investida |
| **3** | Método e robustez (3.1 a 3.5) | 2–3 h | Melhora o que é ensinado, não o que quebra |
| **4** | Documentação | 1–2 h | Depende da Fase 1 para os números virarem verificáveis |
| **5** | Código compartilhado | — | Só se o meio-termo acima convencer |

**Definição de pronto:** `uv sync --all-extras && uv run pytest && MASTER_SCRIPT_FAST=1 uv
run jupyter nbconvert --execute --to notebook 0*.ipynb` passa limpo, em CI, num runner de
2 núcleos.
