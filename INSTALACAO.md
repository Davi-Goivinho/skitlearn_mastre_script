# Instalação das bibliotecas

Ambiente gerenciado com [uv](https://docs.astral.sh/uv/). O `uv` cria o virtualenv,
baixa o Python certo e instala tudo a partir do `pyproject.toml` — não é preciso
`python -m venv`, `pip` nem `requirements.txt`.

---

## 1. Instalar o uv

Já está instalado nesta máquina (`uv 0.11.25`). Em outra máquina:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Conferir:

```bash
uv --version
```

---

## 2. Montar o ambiente

Na raiz do projeto:

```bash
uv sync
```

Isso faz, em uma tacada:

1. lê `.python-version` e garante o **Python 3.14** (baixa se não houver);
2. cria o `.venv/` no diretório do projeto;
3. instala as dependências do `pyproject.toml` **nas versões exatas do `uv.lock`**;
4. instala também o grupo `dev` (JupyterLab, ipykernel, ipywidgets), que vem por padrão.

Não é preciso ativar o venv: use `uv run <comando>`, que resolve o ambiente sozinho.

---

## 3. Abrir os notebooks

```bash
uv run jupyter lab
```

O kernel `Python 3` que aparece no JupyterLab já é o do `.venv`. Se preferir registrar um
kernel com nome próprio, para escolher a partir de outro Jupyter:

```bash
uv run python -m ipykernel install --user --name sklearn-master --display-name "Master Script (3.14)"
```

Para rodar algo sem abrir o Jupyter:

```bash
uv run python meu_script.py
```

---

## 4. O que é instalado por padrão

Tudo que os três notebooks importam na célula de setup:

| Pacote | Papel | Versão travada |
|---|---|---|
| `scikit-learn` | pipelines, modelos, métricas, buscas | 1.9.0 |
| `numpy` | base numérica | 2.4.6 |
| `pandas` | DataFrames | 3.0.5 |
| `scipy` | `loguniform`/`randint`/`uniform` das grades, dendrograma | 1.18.0 |
| `matplotlib` | todos os gráficos | 3.11.1 |
| `joblib` | persistência do modelo (`.joblib`) | 1.5.3 |
| `pyarrow` | leitura e escrita de `.parquet` | 25.0.0 |
| `xgboost` | boosting externo | 3.4.0 |
| `lightgbm` | boosting externo | 4.7.0 |
| `catboost` | boosting externo | 1.2.10 |
| `imbalanced-learn` | SMOTE e `ImbPipeline` (notebook 01) | 0.14.2 |
| `jupyterlab` + `ipykernel` + `ipywidgets` | ambiente de trabalho (grupo `dev`) | 4.6.2 |

O `scikit-learn` está fixado em `>=1.6` porque os notebooks usam `HDBSCAN`,
`root_mean_squared_error`, `TargetEncoder`, `PredictionErrorDisplay`, `class_weight` no
`HistGradientBoosting` e `set_output(transform="pandas")` — recursos que não existem em
versões anteriores.

---

## 5. Pacotes opcionais (extras)

Ficam de fora por padrão. Aparecem só em células comentadas dos notebooks.

| Extra | Pacotes | Para quê |
|---|---|---|
| `explain` | `shap` | Interpretabilidade (células 6.6 do nb 01 e 6.5 do nb 02) |
| `tuning` | `optuna` | Busca bayesiana de hiperparâmetros (células 5.5) |
| `cluster` | `umap-learn`, `kneed`, `kmodes` | Projeção não-linear, cotovelo automático, clusterização categórica (nb 03) |
| `io` | `openpyxl`, `sqlalchemy` | `pd.read_excel`, `pd.read_sql` |

Instalar um:

```bash
uv sync --extra explain
```

Instalar vários:

```bash
uv sync --extra explain --extra cluster
```

Instalar todos:

```bash
uv sync --all-extras
```

> **Estado atual desta máquina:** o ambiente foi sincronizado com `--all-extras`, então
> `shap`, `optuna`, `umap-learn`, `kneed`, `kmodes`, `openpyxl` e `sqlalchemy` já estão
> instalados. Rodar `uv sync` sem flags **remove** os extras, voltando ao conjunto básico.

### Por que o numpy está travado em `<2.5`

Não é capricho. `shap` e `umap-learn` dependem do `numba`, e o `numba` 0.66 — a versão
atual, com suporte a Python 3.14 — ainda exige `numpy<2.5`. Sem esse teto na base, ativar
qualquer um desses extras faz o resolvedor recuar até o `numba` 0.53 (de 2021), que não tem
wheel publicado e falha ao compilar.

Como numpy 2.4 e 2.5 são equivalentes para o que estes notebooks fazem, o teto vale mais a
pena do que perder os extras. **Se você não usa `shap` nem `umap-learn`**, remova o `<2.5`
do `pyproject.toml` e rode `uv lock --upgrade` para subir ao numpy mais recente.

---

## 6. Reprodutibilidade

O `uv.lock` guarda a versão exata e o hash de cada um dos 139 pacotes. Ele **deve** ser
versionado junto com o projeto: é o que faz outra máquina montar o ambiente idêntico.

```bash
uv sync --frozen
```

Instala exatamente o que está no lock e falha se o `pyproject.toml` tiver mudado sem
relock — o modo recomendado em CI e ao retomar o projeto meses depois.

---

## 7. Atualizar as bibliotecas

Subir tudo o que as restrições do `pyproject.toml` permitirem:

```bash
uv lock --upgrade
```

Subir um pacote específico:

```bash
uv lock --upgrade-package scikit-learn
```

Depois de qualquer um dos dois, aplicar:

```bash
uv sync
```

> **Atenção que custou tempo aqui:** por padrão o `uv lock` **preserva** as versões já
> travadas enquanto elas continuarem satisfazendo as restrições — mesmo que exista coisa
> muito mais nova. Editar o `pyproject.toml` e rodar `uv lock` pode deixar um pacote preso
> numa versão antiga sem nenhum aviso. Quando o resultado da resolução não fizer sentido,
> `uv lock --upgrade` é a primeira coisa a tentar.

---

## 8. Adicionar e remover pacotes

```bash
uv add seaborn
```

```bash
uv add --optional cluster hdbscan
```

```bash
uv add --dev pytest
```

```bash
uv remove seaborn
```

Cada comando atualiza `pyproject.toml`, `uv.lock` e o `.venv` de uma vez.

---

## 9. Trocar a versão do Python

```bash
uv python pin 3.13
```

```bash
uv sync
```

O `pyproject.toml` declara `requires-python = ">=3.12"`. O piso vem do **scipy 1.18**, que
exige 3.12 (pandas 3, scikit-learn 1.9 e numpy ainda aceitam 3.11). Manter a faixa fechada
também evita que o resolvedor bifurque a resolução para acomodar 3.11, o que puxa versões
mais antigas de alguns pacotes.

Ver o que há disponível:

```bash
uv python list
```

---

## 10. Sem uv (pip + venv)

Alternativa, caso precise montar o ambiente em outro lugar:

```bash
python3.14 -m venv .venv && source .venv/bin/activate
```

```bash
pip install "scikit-learn>=1.6" "numpy>=1.26,<2.5" "pandas>=2.2" "scipy>=1.13" matplotlib joblib pyarrow xgboost lightgbm catboost imbalanced-learn jupyterlab ipykernel ipywidgets
```

Gerar um `requirements.txt` a partir do lock, com as versões exatas:

```bash
uv export --no-hashes --format requirements-txt > requirements.txt
```

---

## Solução de problemas

**`ModuleNotFoundError` dentro do notebook, mas o pacote está instalado**
O Jupyter está em outro kernel. Feche e abra com `uv run jupyter lab`, ou selecione o
kernel do `.venv` no canto superior direito.

**A instalação tenta compilar algo do zero e falha**
Sinal de que o resolvedor recuou para uma versão antiga sem wheel. Rode
`uv lock --upgrade` e sincronize de novo. Se persistir, veja qual pacote é o gargalo:

```bash
uv tree --package numba
```

**Quero começar do zero**

```bash
rm -rf .venv && uv sync
```

**Ver o que está instalado agora**

```bash
uv pip list
```

**O ambiente ocupa quanto?**
Cerca de **1,6 GB** com todos os extras. A maior parte é `catboost`, `xgboost`, `lightgbm`,
`scipy` e as bibliotecas nativas do `numba`/`llvmlite`.
