"""Verificações do ajuste de corte de decisão no notebook 01 (células 6.5 e 6.5b).

Um sinal trocado no scorer de custo não gera erro: o estimador apenas passa a procurar
o pior corte. Por isso os testes executam as células reais e conferem o resultado.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone

from conftest import NOTEBOOKS, celulas_de_codigo

CELULAS = ("# ==== 6.1", "# ==== 6.5 ", "# ==== 6.5b")


@pytest.fixture(scope="module")
def resultado(ns):
    """Roda as células com uma regressão logística no lugar do vencedor do torneio."""
    conteudo = dict(ns("01"))
    conteudo["best_model"] = clone(conteudo["PIPELINES"]["LogisticRegression"]).fit(
        conteudo["X_train"], conteudo["y_train"]
    )
    fontes = celulas_de_codigo(NOTEBOOKS["01"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for prefixo in CELULAS:
            fonte = next(f for f in fontes if f.startswith(prefixo))
            exec(compile(fonte, prefixo, "exec"), conteudo)
    return conteudo


def test_predict_aplica_o_corte_escolhido(resultado):
    X_test = resultado["X_test"]
    for nome, tuned in resultado["AJUSTADOS"].items():
        proba = tuned.predict_proba(X_test)[:, 1]
        esperado = (proba >= tuned.best_threshold_).astype(int)
        assert np.array_equal(tuned.predict(X_test), esperado), nome


def test_custo_alto_de_fn_baixa_o_corte(resultado):
    """Com FN custando 10x um FP, o corte de custo mínimo fica bem abaixo de 0,5."""
    assert resultado["AJUSTADOS"]["custo mínimo"].best_threshold_ < 0.4


def test_custo_escolhido_e_o_menor_da_curva(resultado):
    """Pega o sinal invertido: o corte precisa ter o menor custo entre os candidatos."""
    tuned = resultado["AJUSTADOS"]["custo mínimo"]
    custos = -tuned.cv_results_["scores"]
    assert (custos >= 0).all()
    assert custos[np.argmax(tuned.cv_results_["thresholds"] == tuned.best_threshold_)] == custos.min()


def test_estimador_e_manual_concordam(resultado):
    """As duas contas do guia medem a mesma coisa e devem chegar perto."""
    tuned = resultado["AJUSTADOS"]["custo mínimo"].best_threshold_
    assert abs(tuned - resultado["thr_custo"]) < 0.1


def test_corte_fixo_nao_retreina_o_modelo(resultado):
    """FrozenEstimator: o modelo interno continua sendo o best_model já ajustado."""
    interno = resultado["modelo_com_corte"].estimator_.estimator
    assert interno is resultado["best_model"]
