"""Verificações estáticas das grades e dos pipelines dos três notebooks.

Rodam em segundos e cobrem a classe de defeito que já custou caro aqui: nome de
scorer inexistente, família sem grade e chave de grade com prefixo errado.
"""

import pytest
from sklearn.base import clone
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import ParameterSampler

TAGS = ["01", "02", "03", "04"]

# Famílias declaradas em PIPELINES que legitimamente não têm grade: baselines e
# modelos sem hiperparâmetro. Qualquer outra ausência é esquecimento e quebra a
# etapa 5 com KeyError quando ela vence o torneio.
SEM_GRADE = {
    "01": {"Dummy"},
    "02": {"Dummy", "LinearRegression"},
    "03": set(),
    "04": {"Dummy", "Persistence", "SeasonalNaive", "LinearRegression"},
}

# Contagem por notebook. Fixada aqui para que o número deixe de ser algo escrito
# à mão na documentação e passe a ser verificável.
N_FAMILIAS = {"01": 20, "02": 27, "03": 14, "04": 22}


@pytest.mark.parametrize("tag", TAGS)
def test_contagem_de_familias(tag, ns):
    assert len(ns(tag)["PIPELINES"]) == N_FAMILIAS[tag]


# O nb 03 monta scorers próprios (funções), porque clusterização não tem y.
# Nos supervisionados o SCORING é feito de nomes do registro do sklearn.
SCORERS_POR_NOME = {"01", "02", "04"}


@pytest.mark.parametrize("tag", TAGS)
def test_scorers_existem(tag, ns):
    """Todo scorer nomeado por string precisa estar no registro do sklearn.

    O `cross_validate` valida o scoring ANTES de qualquer fit, então um nome
    errado não falha um modelo: derruba o torneio inteiro de uma vez.
    """
    scoring = ns(tag).get("SCORING")
    # Sem esta guarda o teste passaria calado se o SCORING deixasse de ser
    # coletado (por mudança de seção, por exemplo) — justo o cenário que ele
    # existe para cobrir.
    assert scoring, f"notebook {tag}: SCORING não foi encontrado ou está vazio"

    nomeados = {k: v for k, v in scoring.items() if isinstance(v, str)}
    if tag in SCORERS_POR_NOME:
        assert nomeados, f"notebook {tag}: nenhum scorer nomeado por string em SCORING"

    validos = set(get_scorer_names())
    invalidos = {k: v for k, v in nomeados.items() if v not in validos}
    assert not invalidos, (
        f"notebook {tag}: scorer inexistente em SCORING -> {invalidos}. "
        f"Métricas de erro são registradas na forma negativa (neg_...)."
    )


@pytest.mark.parametrize("tag", TAGS)
def test_toda_familia_tem_grade(tag, ns):
    conteudo = ns(tag)
    faltando = set(conteudo["PIPELINES"]) - set(conteudo["GRIDS"]) - SEM_GRADE[tag]
    assert not faltando, (
        f"notebook {tag}: famílias em PIPELINES sem entrada em GRIDS -> "
        f"{sorted(faltando)}. Se for de propósito, declare em SEM_GRADE e trate "
        f"o caso na etapa 5."
    )


@pytest.mark.parametrize("tag", TAGS)
def test_grade_orfa(tag, ns):
    conteudo = ns(tag)
    orfas = set(conteudo["GRIDS"]) - set(conteudo["PIPELINES"])
    assert not orfas, f"notebook {tag}: grades sem pipeline -> {sorted(orfas)}"


@pytest.mark.parametrize("tag", TAGS)
def test_set_params_aceita_a_grade(tag, ns):
    """Amostras de cada grade precisam ser aplicáveis ao pipeline correspondente.

    Pega erro de digitação e prefixo errado ("modelo__", "prep__cat__scaler" em
    pipeline que não tem esse passo). Ponto cego conhecido: o CatBoost aceita via
    set_params parâmetros que não aparecem no get_params de uma instância nova —
    eles sobrevivem ao clone e funcionam, mas ali este teste não detecta typo.
    """
    conteudo = ns(tag)
    for nome, grade in conteudo["GRIDS"].items():
        pipe = conteudo["PIPELINES"][nome]
        for params in ParameterSampler(grade, n_iter=5, random_state=0):
            try:
                clone(pipe).set_params(**params)
            except Exception as erro:
                pytest.fail(
                    f"notebook {tag}, grade {nome}: set_params rejeitou "
                    f"{sorted(params)} -> {type(erro).__name__}: {erro}"
                )


@pytest.mark.parametrize("tag", TAGS)
def test_ultimo_passo_se_chama_model(tag, ns):
    """A convenção da qual todas as grades dependem: o prefixo `model__`."""
    fora_do_padrao = [
        nome for nome, pipe in ns(tag)["PIPELINES"].items()
        if pipe.steps[-1][0] != "model"
    ]
    assert not fora_do_padrao, (
        f"notebook {tag}: pipelines cujo último passo não se chama 'model' -> "
        f"{fora_do_padrao}. As grades usam o prefixo model__ e falhariam."
    )


@pytest.mark.parametrize("tag", TAGS)
def test_paralelismo_nao_esta_aninhado(tag, ns):
    """Nenhum estimador pode usar n_jobs=-1: a CV já paraleliza por fora.

    Os dois níveis juntos geram n_núcleos² threads disputando n_núcleos, e o
    LightGBM trava em vez de ficar apenas lento.
    """
    culpados = []
    for nome, pipe in ns(tag)["PIPELINES"].items():
        for chave, valor in pipe.get_params(deep=True).items():
            # thread_count é como o CatBoost chama o n_jobs dele.
            if chave.split("__")[-1] in ("n_jobs", "thread_count") and valor == -1:
                culpados.append(f"{nome}.{chave}")
    assert not culpados, (
        f"notebook {tag}: paralelismo -1 dentro do estimador -> {culpados}. "
        f"Use N_JOBS_MODEL; o -1 fica só na validação cruzada e na busca."
    )
