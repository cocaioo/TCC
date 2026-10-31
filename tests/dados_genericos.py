"""Tabela genérica para os testes: uma matriz numérica artificial, sem nenhum significado clínico.

Serve só para verificar a MECÂNICA do pipeline: os algoritmos treinam, nada vaza entre as dobras, as métricas conferem
com contas feitas à mão. É criada na memória a cada teste, não é gravada no repositório e nenhum número dela vai para
relatório. As variáveis se chamam x0, x1, ...; só o identificador e o desfecho usam os nomes do projeto.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from divisao import COL_ALVO, COL_ID  # noqa: E402


def tabela_generica(n=400, prevalencia=0.20, n_variaveis=8, seed=0):
    """`n` linhas: identificador, `n_variaveis` colunas numéricas (x0, x1, ...) e o desfecho 0/1 com a prevalência dada."""
    X, y = make_classification(n_samples=n, n_features=n_variaveis, n_informative=4, n_redundant=2,
                               weights=[1 - prevalencia], flip_y=0.01, random_state=seed)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_variaveis)])
    df.insert(0, COL_ID, [f"P{i:04d}" for i in range(n)])
    df[COL_ALVO] = y
    return df
