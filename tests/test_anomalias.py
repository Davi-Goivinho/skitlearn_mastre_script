"""Verificações específicas do notebook 05.

Em detecção de anomalias, um sinal invertido não gera erro: o ranking apenas sai ao
contrário. Estes testes conferem o resultado, não só a execução.
"""

import warnings

import pytest
from sklearn.base import clone
from sklearn.metrics import average_precision_score

BASELINE_ALEATORIO = "Aleatorio"


@pytest.fixture(scope="module")
def ajustados(ns):
    """Cada família ajustada uma vez no treino, sem usar o rótulo."""
    conteudo = ns("05")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            nome: clone(pipe).fit(conteudo["X_train"])
            for nome, pipe in conteudo["PIPELINES"].items()
        }


def test_escore_aponta_para_as_anomalias(ajustados, ns):
    """decision_function precisa ser menor nas anomalias, em todas as famílias."""
    conteudo = ns("05")
    X, y = conteudo["X_train"], conteudo["y_train"]
    invertidos = []
    for nome, modelo in ajustados.items():
        if nome == BASELINE_ALEATORIO:
            continue
        decisao = modelo.decision_function(X)
        if decisao[y == 1].mean() >= decisao[y == 0].mean():
            invertidos.append(nome)
    assert not invertidos, f"escore invertido ou sem sinal em: {invertidos}"


def test_dados_tem_sinal_detectavel(ajustados, ns):
    """O melhor detector precisa superar o acaso com folga no teste.

    Protege contra o defeito que já ocorreu no notebook 01: dado sintético sem sinal.
    """
    conteudo = ns("05")
    X, y = conteudo["X_test"], conteudo["y_test"]
    melhor = max(
        average_precision_score(y, -modelo.decision_function(X))
        for nome, modelo in ajustados.items()
        if nome != BASELINE_ALEATORIO
    )
    assert melhor > 3 * y.mean(), f"PR-AUC {melhor:.3f} para prevalência {y.mean():.3f}"


def test_predict_segue_a_convencao(ajustados, ns):
    """Os detectores próprios devem devolver +1 e -1, como os do scikit-learn."""
    X = ns("05")["X_test"]
    for nome, modelo in ajustados.items():
        assert set(modelo.predict(X)) <= {-1, 1}, f"{nome}: predict fora da convenção"
