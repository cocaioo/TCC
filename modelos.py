"""Modelos de aprendizado de máquina do projeto (etapa 3 do KDD: mineração de dados).

Passo 1: catálogo dos 8 algoritmos do Quadro 4 do PDF (GB, XGBoost, DT, KNN, RF, NB, RL, MLP), cada um com duas grades de
hiperparâmetros (enxuta, para testar o pipeline; completa, para o dado real), e a montagem do `Pipeline` de cada um.
Nada aqui ajusta modelo nem roda busca: quem ajusta é a validação cruzada (passo 2).

O pré-processador é sempre o primeiro passo do pipeline ("prep"): assim, a validação cruzada o refaz do zero a cada
dobra e nada é aprendido com a dobra de validação. Os hiperparâmetros das grades levam o prefixo `modelo__` (convenção
do scikit-learn para "o parâmetro do passo chamado modelo").

Os valores das grades são iniciais, a revisar; não foram ajustados ao seed.

Uso:
    python modelos.py --listar
"""

import argparse
import math
import sys

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

MODELOS = ("RL", "NB", "KNN", "DT", "RF", "GB", "XGBoost", "MLP")
GRADES = ("enxuta", "completa")
DESBALANCEAMENTO = ("nenhum", "class_weight")  # "smote" entra no passo 3

# O scikit-learn não tem `class_weight` nestes três: no modo "class_weight" eles rodam sem ajuste.
SEM_PESOS = ("KNN", "NB", "MLP")

# Grades de hiperparâmetros (nomes sem o prefixo `modelo__`, que é acrescentado em `catalogo`).
_GRADES = {
    "RL": {
        "enxuta": {"C": [0.01, 0.1, 1, 10]},
        "completa": {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
    },
    "NB": {
        "enxuta": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "completa": {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
    },
    "KNN": {
        "enxuta": {"n_neighbors": [3, 5, 11, 21], "weights": ["uniform", "distance"]},
        "completa": {"n_neighbors": [3, 5, 7, 11, 15, 21, 31], "weights": ["uniform", "distance"], "p": [1, 2]},
    },
    "DT": {
        "enxuta": {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 5, 10]},
        "completa": {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                     "min_samples_leaf": [1, 3, 5, 10, 20]},
    },
    "RF": {
        "enxuta": {"n_estimators": [100, 300], "max_depth": [None, 8], "min_samples_leaf": [1, 5]},
        "completa": {"n_estimators": [100, 300, 500], "max_depth": [None, 5, 10, 20],
                     "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.3]},
    },
    "GB": {
        "enxuta": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        "completa": {"n_estimators": [100, 200, 400], "learning_rate": [0.01, 0.05, 0.1],
                     "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]},
    },
    "XGBoost": {
        "enxuta": {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "completa": {"n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.05, 0.1],
                     "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]},
    },
    "MLP": {
        "enxuta": {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [1e-4, 1e-2],
                   "learning_rate_init": [1e-3, 1e-2]},
        "completa": {"hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "alpha": [1e-4, 1e-3, 1e-2],
                     "learning_rate_init": [1e-3, 1e-2]},
    },
}


class GradientBoostingBalanceado(GradientBoostingClassifier):
    """Gradient Boosting que pesa cada amostra pelo inverso da frequência da sua classe (como `class_weight="balanced"`).

    O `GradientBoostingClassifier` do scikit-learn não tem `class_weight`, só aceita pesos por amostra no `fit`. Sem
    parâmetro novo no `__init__`, então `clone` e o GridSearchCV funcionam normalmente.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        peso = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            peso = peso * sample_weight
        return super().fit(X, y, sample_weight=peso, **kwargs)


def _estimador(nome, desbalanceamento, prevalencia, seed):
    """Estimador SEM ajuste de um dos 8 modelos, com o ajuste de desbalanceamento pedido (quando o modelo o aceita)."""
    pesos = {"class_weight": "balanced"} if desbalanceamento == "class_weight" else {}
    if nome == "RL":
        return LogisticRegression(max_iter=2000, random_state=seed, **pesos)
    if nome == "NB":
        return GaussianNB()
    if nome == "KNN":
        return KNeighborsClassifier()
    if nome == "DT":
        return DecisionTreeClassifier(random_state=seed, **pesos)
    if nome == "RF":
        return RandomForestClassifier(random_state=seed, **pesos)
    if nome == "GB":
        classe = GradientBoostingBalanceado if pesos else GradientBoostingClassifier
        return classe(random_state=seed)
    if nome == "XGBoost":
        # negativos/positivos: com as dobras estratificadas, a prevalência de cada dobra é praticamente a do treino
        extra = {"scale_pos_weight": (1 - prevalencia) / prevalencia} if pesos else {}
        # n_jobs=1: o paralelismo fica na busca em grade (n_jobs do GridSearchCV); sem isso, as threads do XGBoost competem
        return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, **extra)
    if nome == "MLP":
        return MLPClassifier(max_iter=500, early_stopping=True, random_state=seed)
    raise ValueError(f"modelo desconhecido: {nome!r}")


def construir_pipeline(preprocessador, estimador):
    """`Pipeline([("prep", pré-processador), ("modelo", estimador)])`, ainda sem ajuste.

    O pré-processador é clonado: o `Pipeline` ajusta os passos no próprio objeto, então, sem o clone, os 8 modelos
    compartilhariam o mesmo pré-processador ajustado.
    """
    return Pipeline([("prep", clone(preprocessador)), ("modelo", estimador)])


def catalogo(prevalencia, grade="enxuta", desbalanceamento="nenhum", seed=42):
    """Devolve `{nome: (estimador, grade_de_hiperparametros)}` para os 8 modelos, tudo SEM ajuste.

    `prevalencia` (fração de sepse do treino, `y_treino.mean()`) só é usada no `scale_pos_weight` do XGBoost.
    A grade vem no formato do GridSearchCV, com o prefixo `modelo__` (os estimadores são o passo "modelo" do pipeline).
    """
    if grade not in GRADES:
        raise ValueError(f"grade deve ser uma de {GRADES}, recebi {grade!r}")
    if desbalanceamento not in DESBALANCEAMENTO:
        raise ValueError(f"desbalanceamento deve ser um de {DESBALANCEAMENTO}, recebi {desbalanceamento!r}")
    if desbalanceamento == "class_weight" and not 0 < prevalencia < 1:
        raise ValueError(f"prevalencia deve estar entre 0 e 1 (exclusivo), recebi {prevalencia!r}")
    return {
        nome: (_estimador(nome, desbalanceamento, prevalencia, seed),
               {f"modelo__{parametro}": list(valores) for parametro, valores in _GRADES[nome][grade].items()})
        for nome in MODELOS
    }


def n_combinacoes(grade_de_hiperparametros):
    """Número de combinações de uma grade (cada uma é treinada uma vez por dobra)."""
    return math.prod(len(valores) for valores in grade_de_hiperparametros.values())


def imprimir_catalogo():
    enxuta = catalogo(0.20, "enxuta")
    completa = catalogo(0.20, "completa")
    print(f"{'modelo':<9}{'enxuta':>8}{'completa':>10}  ajuste de desbalanceamento (modo class_weight)")
    for nome in MODELOS:
        if nome in SEM_PESOS:
            ajuste = "nenhum (o scikit-learn não tem class_weight)"
        elif nome == "XGBoost":
            ajuste = "scale_pos_weight = negativos/positivos"
        elif nome == "GB":
            ajuste = "pesos por amostra (GradientBoostingBalanceado)"
        else:
            ajuste = 'class_weight="balanced"'
        print(f"{nome:<9}{n_combinacoes(enxuta[nome][1]):>8}{n_combinacoes(completa[nome][1]):>10}  {ajuste}")
    total_e = sum(n_combinacoes(g) for _, g in enxuta.values())
    total_c = sum(n_combinacoes(g) for _, g in completa.values())
    print(f"{'total':<9}{total_e:>8}{total_c:>10}")
    # validacao.py roda a busca simples (5 dobras) e a aninhada (5 dobras externas x 5 internas); os refits dos
    # vencedores (6 por modelo) não entram na conta
    print(f"\najustes só da busca simples (5 dobras): {5 * total_e} (enxuta) e {5 * total_c} (completa)")
    print(f"ajustes de `validacao.py` (busca simples + aninhada 5x5 = 30 por combinação): "
          f"{30 * total_e} (enxuta) e {30 * total_c} (completa)")


def main():
    parser = argparse.ArgumentParser(description="Catálogo de modelos do projeto de sepse neonatal tardia.")
    parser.add_argument("--listar", action="store_true", help="imprime o catálogo (modelos, grades e ajuste de classe)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows usa cp1252 e quebra os acentos
    if args.listar:
        imprimir_catalogo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
