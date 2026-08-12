"""Executa cada notebook de ponta a ponta, em modo rápido.

É a única verificação que pega travamento e erro em cascata — as duas coisas que
os testes estáticos não veem. Marcada como `lento` e fora da rodada padrão:

    uv run pytest                          # só os estáticos, segundos
    MASTER_SCRIPT_FAST=1 uv run pytest -m lento
"""

import os

import nbformat
import pytest
from nbclient import NotebookClient

from conftest import NOTEBOOKS

TEMPO_LIMITE_POR_CELULA = 900   # generoso: o torneio é a célula mais cara


@pytest.mark.lento
@pytest.mark.parametrize("tag", sorted(NOTEBOOKS))
def test_notebook_executa_de_ponta_a_ponta(tag, tmp_path):
    os.environ["MASTER_SCRIPT_FAST"] = "1"

    nb = nbformat.read(NOTEBOOKS[tag], as_version=4)
    cliente = NotebookClient(
        nb,
        timeout=TEMPO_LIMITE_POR_CELULA,
        kernel_name="python3",
        # Roda com o cwd no tmp_path: os artefatos das etapas 6 (.joblib,
        # base_segmentada.parquet, catboost_info/) não sujam a raiz do projeto.
        resources={"metadata": {"path": str(tmp_path)}},
    )
    cliente.execute()   # levanta CellExecutionError na primeira célula que falhar
