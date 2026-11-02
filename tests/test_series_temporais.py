"""Verificações específicas do notebook 04."""

import numpy as np
import pandas as pd
from sklearn.base import clone


def test_linha_futura_reproduz_a_engenharia_de_variaveis(ns):
    """A previsão multi-passo só vale se o futuro for montado como o treino foi.

    Se uma variável mudar em 1.3 e não em linha_futura, o modelo passa a receber
    outra coisa em produção, sem erro nenhum.
    """
    conteudo = ns("04")
    X, df, bruto = conteudo["X"], conteudo["df"], conteudo["df_raw"].sort_values("data")
    valores = bruto["demanda"].to_numpy()
    deslocamento = len(bruto) - len(df)

    for posicao in (0, 50, len(df) - 1):
        o = posicao + deslocamento
        linha = conteudo["linha_futura"](
            valores[:o], df["data"].iloc[posicao], o, df["promocao"].iloc[posicao]
        )
        esperado = X.iloc[[posicao]].reset_index(drop=True)
        pd.testing.assert_frame_equal(linha, esperado, check_dtype=False)


def test_hibrido_participa_do_torneio_e_preve(ns):
    conteudo = ns("04")
    assert "Hibrido" in conteudo["PIPELINES"]
    modelo = clone(conteudo["PIPELINES"]["Hibrido"]).fit(conteudo["X_train"], conteudo["y_train"])
    assert np.isfinite(modelo.predict(conteudo["X_test"])).all()


def test_quantile_nao_colapsa_na_constante(ns):
    """Com o alpha padrão, os coeficientes zeravam e o modelo virava o Dummy."""
    conteudo = ns("04")
    modelo = clone(conteudo["PIPELINES"]["QuantileRegressor"]).fit(
        conteudo["X_train"], conteudo["y_train"]
    )
    assert np.unique(modelo.predict(conteudo["X_test"]).round(6)).size > 1
