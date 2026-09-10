"""Infraestrutura comum dos testes.

O truque central está em `namespace()`: ela executa um notebook só até onde
interessa — a célula que declara `GRIDS` — pulando a seção de torneio, que roda
toda a validação cruzada e levaria minutos. O que sobra é barato e dá acesso aos
objetos reais (`PIPELINES`, `GRIDS`, `SCORING`) para as asserções.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent

NOTEBOOKS = {
    "01": RAIZ / "01_classificacao.ipynb",
    "02": RAIZ / "02_regressao.ipynb",
    "03": RAIZ / "03_clusterizacao.ipynb",
    "04": RAIZ / "04_series_temporais.ipynb",
}

# A seção 4 é o torneio: roda cross_validate em todas as famílias. Nada dela é
# necessário para validar as grades. A 4.1 fica de fora da lista de propósito —
# é ela que declara SCORING, justamente o que precisa ser testado.
SECOES_PULADAS = ("# ==== 4.2", "# ==== 4.3", "# ==== 4.4", "# ==== 4.5")

# A partir daqui não há mais nada a coletar: GRIDS é o último objeto que os
# testes usam, e o que vem depois (busca, prova final) é caro.
MARCA_DE_PARADA = "GRIDS = {"


def celulas_de_codigo(caminho):
    with open(caminho, encoding="utf-8") as fh:
        nb = json.load(fh)
    return [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    ]


@lru_cache(maxsize=None)
def namespace(tag):
    """Executa o notebook `tag` até a declaração de GRIDS e devolve o namespace."""
    import matplotlib

    matplotlib.use("Agg")

    os.environ["MASTER_SCRIPT_FAST"] = "1"   # 300 linhas em vez de 1.000+
    ns = {"__name__": "__main__", "display": lambda *a, **k: None}

    for i, fonte in enumerate(celulas_de_codigo(NOTEBOOKS[tag])):
        if fonte.lstrip().startswith(SECOES_PULADAS):
            continue
        try:
            exec(compile(fonte, f"{tag}:celula{i}", "exec"), ns)
        except Exception as erro:                      # pragma: no cover
            raise RuntimeError(
                f"notebook {tag}, célula {i}: {type(erro).__name__}: {erro}"
            ) from erro
        if MARCA_DE_PARADA in fonte:
            return ns

    raise RuntimeError(f"notebook {tag}: não achei a declaração {MARCA_DE_PARADA!r}")


@pytest.fixture(scope="session")
def ns():
    """Fixture que devolve a própria função: `ns("01")["PIPELINES"]`."""
    return namespace
