"""Validação dos modelos (etapa 3 do KDD, passo 2): busca em grade e validação cruzada, só com o conjunto de TREINO.

Para cada modelo do catálogo (`modelos.py`), duas estimativas de desempenho:

  busca simples  `GridSearchCV` (5 dobras) no treino inteiro. A nota do vencedor é um pouco OTIMISTA: ele foi escolhido
                 por ser o melhor justamente nessas dobras.
  aninhada       validação cruzada EXTERNA (5 dobras) em volta da busca: em cada rodada, a busca roda só nas 4 dobras
                 de estudo e o melhor modelo é avaliado na dobra que ficou de fora. É a nota honesta, e as
                 probabilidades dessa dobra (fora da dobra) são as que alimentam as curvas ROC do passo 4.

O conjunto de teste NÃO é usado aqui (nem sequer é lido). Os 8 modelos recebem as mesmas dobras externas.

Uso:
    python validacao.py --grade enxuta --desbalanceamento class_weight
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, f1_score, make_scorer, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modelos import DESBALANCEAMENTO, GRADES, MODELOS, catalogo, construir_pipeline
from preprocessamento import COL_ALVO, COL_ID, carregar, construir_preprocessador, separar_x_y

N_DOBRAS = 5  # k-fold com k = 5 (p. 46 do PDF), na busca e na validação externa

# F1, precisão e recall com zero_division=0: uma dobra sem nenhum positivo previsto vale 0, e não um aviso.
# `roc_auc` e `accuracy` usam os nomes do scikit-learn.
METRICAS = {
    "f1": make_scorer(f1_score, zero_division=0),
    "precisao": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "acuracia": "accuracy",
    "auroc": "roc_auc",
}


def _escolher_melhor(cv_results):
    """Índice da melhor combinação: maior F1 médio; empate de F1 (comum quando todos valem 0) é desfeito pela AUROC
    média; empate nos dois fica com a primeira da grade. Usado como `refit` do GridSearchCV."""
    f1 = np.asarray(cv_results["mean_test_f1"])
    auroc = np.asarray(cv_results["mean_test_auroc"])
    empatados = np.flatnonzero(f1 >= f1.max() - 1e-12)
    return int(empatados[np.argmax(auroc[empatados])])


def _busca(pipeline, grade, n_dobras, seed, n_jobs):
    """GridSearchCV pelo F1. `error_score="raise"`: um ajuste que falha para a execução, em vez de virar NaN calado."""
    cv = StratifiedKFold(n_splits=n_dobras, shuffle=True, random_state=seed)
    return GridSearchCV(pipeline, grade, scoring=METRICAS, refit=_escolher_melhor, cv=cv, n_jobs=n_jobs,
                        error_score="raise")


def _preparar(X, y, n_dobras):
    """Confere e normaliza as entradas: X DataFrame, y como Series 0/1 (posicional) e positivos suficientes.

    Com poucos exemplos de uma classe a execução NÃO falha sozinha (F1 = 0 e AUROC ~ 0,5 viram resultado "normal"), por
    isso o mínimo de `n_dobras**2` exemplos por classe é exigido aqui: dá pelo menos alguns por dobra da busca interna.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X deve ser um DataFrame (o pré-processador usa os nomes das colunas)")
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    if len(y) != len(X):
        raise ValueError(f"X e y têm números de linhas diferentes ({len(X)} e {len(y)})")
    if not set(y.unique()) <= {0, 1}:
        raise ValueError(f"y deve conter só 0 e 1, recebi {sorted(map(str, y.unique()))}")
    y = y.astype(int)
    minimo = n_dobras ** 2
    if min(int(y.sum()), int((y == 0).sum())) < minimo:
        raise ValueError(f"cada classe precisa de pelo menos {minimo} exemplos ({n_dobras} dobras ao quadrado); "
                         f"há {int(y.sum())} positivos e {int((y == 0).sum())} negativos")
    return X.reset_index(drop=True), y


def _metricas(y_real, previsao, proba):
    return {
        "f1": f1_score(y_real, previsao, zero_division=0),
        "precisao": precision_score(y_real, previsao, zero_division=0),
        "recall": recall_score(y_real, previsao, zero_division=0),
        "acuracia": accuracy_score(y_real, previsao),
        "auroc": roc_auc_score(y_real, proba),
    }


def avaliar_modelo(pipeline, grade, X, y, n_dobras=N_DOBRAS, seed=42, n_jobs=1):
    """Busca simples e validação aninhada de UM modelo. `X` e `y` são o treino (DataFrame e Series).

    Devolve `{"simples": {...}, "aninhada": {...}, "oof": array}`; `oof[i]` é a probabilidade de sepse do paciente `i`
    prevista por um modelo que NÃO o viu (nem na busca de hiperparâmetros).
    """
    X, y = _preparar(X, y, n_dobras)

    # --- busca simples: nota otimista do vencedor
    busca = _busca(pipeline, grade, n_dobras, seed, n_jobs).fit(X, y)
    i = busca.best_index_
    simples = {f"{m}": busca.cv_results_[f"mean_test_{m}"][i] for m in METRICAS}
    simples["params"] = busca.best_params_

    # --- aninhada: a busca inteira roda dentro de cada dobra externa
    externa = StratifiedKFold(n_splits=n_dobras, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    por_dobra, melhores = [], []
    for treino, validacao in externa.split(X, y):
        interna = _busca(clone(pipeline), grade, n_dobras, seed, n_jobs).fit(X.iloc[treino], y.iloc[treino])
        proba = interna.predict_proba(X.iloc[validacao])[:, 1]
        oof[validacao] = proba
        por_dobra.append(_metricas(y.iloc[validacao], interna.predict(X.iloc[validacao]), proba))
        melhores.append(interna.best_params_)
    tabela = pd.DataFrame(por_dobra)
    # O desvio (ddof=1) resume a variação entre as dobras externas; NÃO é um intervalo de confiança do erro de
    # generalização, porque as dobras compartilham a maior parte do treino.
    aninhada = {"media": tabela.mean().to_dict(), "desvio": tabela.std(ddof=1).to_dict(), "melhores": melhores,
                "por_dobra": tabela}
    return {"simples": simples, "aninhada": aninhada, "oof": oof}


def comparar_modelos(X, y, ids=None, grade="enxuta", desbalanceamento="class_weight", modelos=None,
                     n_dobras=N_DOBRAS, seed=42, n_jobs=-1, mini_catalogo=None, verbose=True):
    """Avalia os modelos pedidos (padrão: os 8) e devolve `(tabela, oof)`.

    tabela  uma linha por modelo: métricas da validação aninhada (média e desvio das dobras externas), a nota otimista
            da busca simples e os melhores hiperparâmetros da busca simples;
    oof     DataFrame com `id_paciente`, o alvo e a probabilidade fora da dobra de cada modelo.

    O pré-processador é construído UMA vez, mas só os NOMES das colunas vêm daqui: tudo o que depende de valores (colunas
    binárias, exames com indicador de ausência, medianas, escalas, pesos de classe) é decidido de novo no `fit` de cada
    dobra.
    """
    X, y = _preparar(X, y, n_dobras)
    if ids is not None and len(ids) != len(y):
        raise ValueError(f"ids e y têm números de linhas diferentes ({len(ids)} e {len(y)})")
    cat = mini_catalogo if mini_catalogo is not None else catalogo(grade, desbalanceamento, seed)
    if isinstance(modelos, str):
        modelos = [modelos]
    escolhidos = list(dict.fromkeys(modelos)) if modelos else list(cat)  # sem repetidos, na ordem dada
    desconhecidos = [m for m in escolhidos if m not in cat]
    if desconhecidos:
        raise ValueError(f"modelos desconhecidos: {desconhecidos}; os disponíveis são {list(cat)}")

    prep = construir_preprocessador(X)
    linhas = []
    oof = pd.DataFrame({COL_ID: np.asarray(ids) if ids is not None else np.arange(len(y)), COL_ALVO: y.to_numpy()})
    for nome in escolhidos:
        estimador, grade_hiper = cat[nome]
        inicio = time.time()
        r = avaliar_modelo(construir_pipeline(prep, estimador), grade_hiper, X, y, n_dobras, seed, n_jobs)
        segundos = time.time() - inicio
        media, desvio = r["aninhada"]["media"], r["aninhada"]["desvio"]
        linhas.append({
            "modelo": nome,
            "f1": media["f1"], "f1_desvio": desvio["f1"], "auroc": media["auroc"], "precisao": media["precisao"],
            "recall": media["recall"], "acuracia": media["acuracia"],
            "f1_busca_simples": r["simples"]["f1"],
            "otimismo_f1": r["simples"]["f1"] - media["f1"],
            "melhores_params": r["simples"]["params"], "segundos": segundos,
        })
        oof[nome] = r["oof"]
        if verbose:
            print(f"  {nome:<8} F1 aninhado {media['f1']:.3f} | AUROC {media['auroc']:.3f} | {segundos:.0f} s", flush=True)
    tabela = pd.DataFrame(linhas).set_index("modelo")
    return tabela, oof


def imprimir_tabela(tabela):
    colunas = ["f1", "f1_desvio", "auroc", "precisao", "recall", "acuracia", "f1_busca_simples", "otimismo_f1", "segundos"]
    cabecalho = {"f1": "F1", "f1_desvio": "dp(F1)", "auroc": "AUROC", "precisao": "precisão", "recall": "recall",
                 "acuracia": "acurácia", "f1_busca_simples": "F1 simples", "otimismo_f1": "otimismo", "segundos": "seg"}
    print(tabela[colunas].rename(columns=cabecalho).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nmelhores hiperparâmetros (busca simples):")
    for nome, params in tabela["melhores_params"].items():
        print(f"  {nome:<8} {params}")


def main():
    parser = argparse.ArgumentParser(description="Busca em grade e validação cruzada dos modelos (só o treino).")
    parser.add_argument("--treino", type=Path, default=Path("data/treino.csv"))
    parser.add_argument("--saida-dir", type=Path, default=Path("data/resultados"))
    parser.add_argument("--grade", choices=GRADES, default="enxuta")
    parser.add_argument("--desbalanceamento", choices=DESBALANCEAMENTO, default="class_weight")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS, help="subconjunto de modelos (padrão: os 8)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="núcleos da busca em grade (-1 = todos)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows usa cp1252 e quebra os acentos
    # Categoria rara que cai só na dobra de validação vira "zeros" no one-hot (handle_unknown="ignore", por desenho) e o
    # scikit-learn avisa a cada dobra. O filtro vale no processo principal e é repassado aos workers pelo scikit-learn
    # (uma variável PYTHONWARNINGS definida aqui não teria efeito: o interpretador já foi iniciado).
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    df = carregar(args.treino)
    X, y = separar_x_y(df)
    print(f"treino: {len(df)} pacientes, {int(y.sum())} com sepse ({y.mean():.1%}); grade {args.grade}; "
          f"desbalanceamento {args.desbalanceamento}\n(o conjunto de teste não é lido)\n")
    tabela, oof = comparar_modelos(X, y, df[COL_ID], args.grade, args.desbalanceamento, args.modelos,
                                   seed=args.seed, n_jobs=args.n_jobs)
    print()
    imprimir_tabela(tabela)

    args.saida_dir.mkdir(parents=True, exist_ok=True)
    sufixo = f"{args.grade}_{args.desbalanceamento}"
    tabela.to_csv(args.saida_dir / f"comparacao_{sufixo}.csv", encoding="utf-8")
    oof.to_csv(args.saida_dir / f"oof_{sufixo}.csv", index=False, encoding="utf-8")
    print(f"\ngravado em {args.saida_dir}: comparacao_{sufixo}.csv e oof_{sufixo}.csv")


if __name__ == "__main__":
    main()
