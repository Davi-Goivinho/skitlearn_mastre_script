"""Verificações dos intervalos conformais (células 6.4b dos notebooks 02 e 04).

Um intervalo errado não gera erro: ele só promete uma cobertura que não entrega. Por
isso os testes conferem a cobertura e, na janela móvel, se o futuro não vaza.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone

from conftest import NOTEBOOKS, celulas_de_codigo


def executar(tag, conteudo, modelo, prefixos):
    """Roda as células reais, com um modelo rápido no lugar do vencedor do torneio."""
    conteudo = dict(conteudo)
    conteudo["best_model"] = clone(conteudo["PIPELINES"][modelo]).fit(
        conteudo["X_train"], conteudo["y_train"]
    )
    conteudo["y_pred"] = conteudo["best_model"].predict(conteudo["X_test"])
    fontes = celulas_de_codigo(NOTEBOOKS[tag])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for prefixo in prefixos:
            fonte = next(f for f in fontes if f.startswith(prefixo))
            exec(compile(fonte, prefixo, "exec"), conteudo)
    return conteudo


@pytest.fixture(scope="module")
def regressao(ns):
    return executar("02", ns("02"), "Ridge", ("# ==== 6.4 ", "# ==== 6.4b"))


@pytest.fixture(scope="module")
def serie(ns):
    return executar("04", ns("04"), "BayesianRidge", ("# ==== 6.4 ", "# ==== 6.4b"))


def test_02_bordas_estao_em_ordem(regressao):
    for nome, (inferior, superior) in regressao["intervalos"].items():
        assert (inferior <= superior).all(), nome


def test_02_cv_plus_entrega_a_cobertura(regressao):
    """Tolerância larga: no modo rápido o teste tem só 60 linhas."""
    assert regressao["tabela_intervalos"].loc["CV+ absoluto", "cobertura"] >= 0.75


def test_02_margem_absoluta_e_fixa_e_a_adaptativa_nao(regressao):
    tabela = regressao["tabela_intervalos"]
    baixa, alta = "largura previsão baixa", "largura previsão alta"
    fixa = tabela.loc["CV+ absoluto", alta] / tabela.loc["CV+ absoluto", baixa]
    adaptativa = tabela.loc["CV+ gamma", alta] / tabela.loc["CV+ gamma", baixa]
    assert fixa == pytest.approx(1, abs=0.05)
    assert adaptativa > 1.2


def test_04_janela_nao_olha_o_futuro(serie):
    """Mudar o valor real do dia t não pode alterar a banda do próprio dia t."""
    banda = serie["banda_por_janela"]
    previsao = np.zeros(10)
    reais = np.arange(1.0, 11.0)
    alterados = reais.copy()
    alterados[5] = 1000.0
    # Confiança de 0,99 faz a margem ser o MAIOR erro da janela. Com 0,8, um único valor
    # extremo não muda o quantil, e o teste passaria mesmo com vazamento.
    base = banda(np.ones(20), previsao, reais, 20, 0.99)
    outra = banda(np.ones(20), previsao, alterados, 20, 0.99)
    np.testing.assert_array_equal(base[1][:6], outra[1][:6])
    assert not np.array_equal(base[1][6:], outra[1][6:])


def test_04_janela_entrega_a_cobertura_em_ruido_estavel(serie):
    rng = np.random.default_rng(0)
    reais = rng.normal(size=5000)
    previsao = np.zeros(5000)
    inferior, superior = serie["banda_por_janela"](
        np.abs(rng.normal(size=60)), previsao, reais, 60, 0.8
    )
    cobertura = ((reais >= inferior) & (reais <= superior)).mean()
    assert cobertura == pytest.approx(0.8, abs=0.03)


def test_04_janela_alarga_depois_do_choque(serie):
    tabela = serie["tabela_intervalos"]
    janela = next(n for n in tabela.index if n.startswith("Janela"))
    assert tabela.loc[janela, "largura após o choque"] > tabela.loc["EnbPI", "largura após o choque"]
