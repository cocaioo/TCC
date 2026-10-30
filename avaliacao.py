"""Avaliação final (etapa 4 do KDD): o conjunto de TESTE é aberto aqui, uma única vez.

Entradas, geradas por `validacao.py` só com o treino (uma rodada por modo de desbalanceamento):
  comparacao_<grade>_<modo>.csv          métricas da validação aninhada de cada modelo
  oof_<grade>_<modo>.csv                 probabilidades fora da dobra (curvas ROC do treino)
  modelos_<grade>_<modo>/<modelo>.joblib vencedores da busca simples, reajustados no treino inteiro

Ordem, que é o que protege o teste:
  1. o vencedor (modelo x modo) é escolhido pelo F1 da validação aninhada, desempate pela AUROC, ANTES de ler o teste;
  2. o teste é lido e todos os modelos salvos são medidos nele (tabela; a escolha já está feita);
  3. intervalo de 95% por bootstrap para as métricas do vencedor no teste;
  4. curvas ROC (fora da dobra e teste), uma figura por modo, e matriz de confusão do vencedor.

Limiar de decisão: 0,5 (o `predict` dos modelos), como na validação.

Uso:
    python avaliacao.py --grade enxuta
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from modelos import DESBALANCEAMENTO, GRADES, MODELOS
from preprocessamento import COL_ALVO, carregar, separar_x_y
from validacao import _metricas

matplotlib.use("Agg")  # só grava arquivos; não abre janela
import matplotlib.pyplot as plt  # noqa: E402

METRICAS_TESTE = ("f1", "precisao", "recall", "especificidade", "acuracia", "auroc")

# Paleta categórica validada (ordem fixa: a cor acompanha o modelo, nunca a posição no ranking).
CORES = dict(zip(MODELOS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")))
TINTA, TINTA_SECUNDARIA, SUPERFICIE = "#0b0b0b", "#52514e", "#fcfcfb"


# --------------------------------------------------------------------------------------
# Entradas da validação
# --------------------------------------------------------------------------------------


def _sufixo(grade, modo):
    return f"{grade}_{modo}"


def modos_disponiveis(resultados_dir, grade):
    """Modos de desbalanceamento com comparação e modelos salvos para a grade pedida (na ordem de DESBALANCEAMENTO).

    Comparação sem a pasta de modelos (por exemplo, de uma versão anterior de `validacao.py`) é erro, e não um modo
    excluído em silêncio da escolha do vencedor."""
    resultados_dir = Path(resultados_dir)
    modos = []
    for m in DESBALANCEAMENTO:
        if not (resultados_dir / f"comparacao_{_sufixo(grade, m)}.csv").exists():
            continue
        if not (resultados_dir / f"modelos_{_sufixo(grade, m)}").is_dir():
            raise FileNotFoundError(f"comparacao_{_sufixo(grade, m)}.csv existe, mas modelos_{_sufixo(grade, m)}/ não: "
                                    f"rode validacao.py --desbalanceamento {m} de novo")
        modos.append(m)
    return modos


def ler_comparacoes(resultados_dir, grade, modos):
    """`{modo: tabela da validação}`; a tabela tem uma linha por modelo, indexada pelo nome."""
    return {m: pd.read_csv(Path(resultados_dir) / f"comparacao_{_sufixo(grade, m)}.csv", index_col="modelo")
            for m in modos}


def escolher_vencedor(comparacoes):
    """`(modelo, modo)` com o maior F1 médio da validação aninhada; empate de F1 desfeito pela AUROC; empate nos dois
    fica com o primeiro na ordem de DESBALANCEAMENTO e MODELOS. Usa só resultados do treino."""
    linhas = [(modo, modelo, linha["f1"], linha["auroc"])
              for modo, tabela in comparacoes.items() for modelo, linha in tabela.iterrows()]
    if not linhas:
        raise ValueError("nenhuma comparação disponível: rode validacao.py antes")
    ordem = {m: i for i, m in enumerate(DESBALANCEAMENTO)} | {m: i for i, m in enumerate(MODELOS)}
    linhas.sort(key=lambda t: (-t[2], -t[3], ordem.get(t[0], 99), ordem.get(t[1], 99)))
    modo, modelo, _, _ = linhas[0]
    return modelo, modo


def carregar_modelo(resultados_dir, grade, modo, modelo):
    return joblib.load(Path(resultados_dir) / f"modelos_{_sufixo(grade, modo)}" / f"{modelo}.joblib")


# --------------------------------------------------------------------------------------
# Métricas no teste
# --------------------------------------------------------------------------------------


def medir(pipeline, X, y):
    """Métricas do teste com o limiar do próprio modelo (`predict`); devolve `(métricas, probabilidades)`."""
    proba = pipeline.predict_proba(X)[:, 1]
    return _metricas(y, pipeline.predict(X), proba), proba


def bootstrap(y, previsao, proba, n=1000, seed=42):
    """Intervalo de 95% (percentis 2,5 e 97,5) de cada métrica, reamostrando os pacientes do teste com reposição.

    Reamostragens sem as duas classes são descartadas (a AUROC não existe nelas). Devolve `{métrica: (inf, sup)}`.
    """
    y, previsao, proba = np.asarray(y), np.asarray(previsao), np.asarray(proba)
    rng = np.random.default_rng(seed)
    valores = []
    while len(valores) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        valores.append(_metricas(y[i], previsao[i], proba[i]))
    tabela = pd.DataFrame(valores)
    return {m: (float(tabela[m].quantile(0.025)), float(tabela[m].quantile(0.975))) for m in tabela.columns}


def avaliar(resultados_dir, caminho_teste, grade="enxuta", n_bootstrap=1000, seed=42):
    """Executa a avaliação final e devolve um dicionário com tudo o que é gravado e impresso.

    Chaves: vencedor (modelo, modo), modos, comparacoes, tabela_teste (uma linha por modelo x modo), probas_teste
    ({modo: {modelo: probabilidades}}), y_teste, ic (do vencedor), confusao (do vencedor).
    """
    modos = modos_disponiveis(resultados_dir, grade)
    if not modos:
        raise FileNotFoundError(f"nenhum resultado da grade {grade!r} em {resultados_dir}: rode validacao.py antes")
    comparacoes = ler_comparacoes(resultados_dir, grade, modos)
    modelo_v, modo_v = escolher_vencedor(comparacoes)  # escolhido ANTES de o teste ser lido

    teste = carregar(caminho_teste)
    X, y = separar_x_y(teste)
    linhas, probas = [], {}
    for modo in modos:
        probas[modo] = {}
        for modelo in comparacoes[modo].index:
            metricas, proba = medir(carregar_modelo(resultados_dir, grade, modo, modelo), X, y)
            probas[modo][modelo] = proba
            linhas.append({"modo": modo, "modelo": modelo, **metricas,
                           "f1_validacao": comparacoes[modo].loc[modelo, "f1"],
                           "vencedor": (modelo, modo) == (modelo_v, modo_v)})
    tabela = pd.DataFrame(linhas)[["modo", "modelo", *METRICAS_TESTE, "f1_validacao", "vencedor"]]

    proba_v = probas[modo_v][modelo_v]
    previsao_v = (proba_v > 0.5).astype(int)  # = predict de um classificador binário
    return {
        "vencedor": (modelo_v, modo_v), "modos": modos, "comparacoes": comparacoes, "tabela_teste": tabela,
        "probas_teste": probas, "y_teste": y.to_numpy(),
        "ic": bootstrap(y, previsao_v, proba_v, n_bootstrap, seed),
        "confusao": confusion_matrix(y, previsao_v, labels=[0, 1]),
    }


# --------------------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------------------


def _eixos(titulo):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    fig.patch.set_facecolor(SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    ax.set_title(titulo, color=TINTA, fontsize=11, loc="left")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_SECUNDARIA)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)
    return fig, ax


def figura_roc(y, probas, titulo, caminho):
    """Curvas ROC de vários modelos numa figura; a legenda traz a AUROC de cada um (identidade não fica só na cor)."""
    fig, ax = _eixos(titulo)
    ax.plot([0, 1], [0, 1], color=TINTA_SECUNDARIA, lw=1, ls=":", label="Acaso (AUROC 0,500)")
    estilos = ("-", "--")
    for i, (modelo, proba) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y, proba)
        auroc = roc_auc_score(y, proba)
        ax.plot(fpr, tpr, color=CORES.get(modelo, TINTA), lw=2, ls=estilos[i % 2],
                label=f"{modelo} ({auroc:.3f})".replace(".", ","))
    ax.set_xlabel("1 − especificidade (taxa de falsos positivos)", color=TINTA, fontsize=9)
    ax.set_ylabel("Sensibilidade (taxa de verdadeiros positivos)", color=TINTA, fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(color="#e6e5e0", lw=0.6)
    legenda = ax.legend(loc="lower right", fontsize=8, frameon=False, title="Modelo (AUROC)", title_fontsize=8)
    for texto in legenda.get_texts():
        texto.set_color(TINTA)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SUPERFICIE)
    plt.close(fig)


def figura_confusao(matriz, titulo, caminho):
    fig, ax = _eixos(titulo)
    fig.set_size_inches(4.6, 4.0)
    ax.imshow(matriz, cmap="Blues", vmin=0, vmax=max(1, matriz.max()) * 1.4)
    rotulos = ["Sem sepse", "Sepse"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    ax.set_xlabel("Previsto", color=TINTA, fontsize=9)
    ax.set_ylabel("Real", color=TINTA, fontsize=9)
    for (i, j), valor in np.ndenumerate(matriz):
        ax.text(j, i, str(valor), ha="center", va="center", color=TINTA, fontsize=12)
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SUPERFICIE)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Saída
# --------------------------------------------------------------------------------------


def gravar(r, resultados_dir, grade):
    """Grava a tabela do teste, o resumo do vencedor (JSON), as curvas ROC e a matriz de confusão."""
    resultados_dir = Path(resultados_dir)
    r["tabela_teste"].to_csv(resultados_dir / f"avaliacao_teste_{grade}.csv", index=False, encoding="utf-8")
    for modo in r["modos"]:
        oof = pd.read_csv(resultados_dir / f"oof_{_sufixo(grade, modo)}.csv")
        modelos = list(r["comparacoes"][modo].index)
        figura_roc(oof[COL_ALVO], {m: oof[m] for m in modelos},
                   f"Curvas ROC fora da dobra (treino) · desbalanceamento: {modo}",
                   resultados_dir / f"roc_oof_{_sufixo(grade, modo)}.png")
        figura_roc(r["y_teste"], r["probas_teste"][modo], f"Curvas ROC no teste · desbalanceamento: {modo}",
                   resultados_dir / f"roc_teste_{_sufixo(grade, modo)}.png")
    modelo, modo = r["vencedor"]
    figura_confusao(r["confusao"], f"{modelo} ({modo}) no teste, limiar 0,5",
                    resultados_dir / f"confusao_vencedor_{grade}.png")

    linha = r["tabela_teste"].query("vencedor").iloc[0]
    resumo = {
        "modelo": modelo, "modo": modo, "grade": grade, "limiar": 0.5,
        "validacao_aninhada": {k: float(v) for k, v in r["comparacoes"][modo].loc[modelo].items()
                               if k in METRICAS_TESTE or k == "f1_desvio"},
        "teste": {m: float(linha[m]) for m in METRICAS_TESTE},
        "teste_ic95": {m: list(v) for m, v in r["ic"].items()},
        "confusao": {"vn": int(r["confusao"][0, 0]), "fp": int(r["confusao"][0, 1]),
                     "fn": int(r["confusao"][1, 0]), "vp": int(r["confusao"][1, 1])},
    }
    (resultados_dir / f"vencedor_{grade}.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")
    return resumo


def imprimir(r):
    modelo, modo = r["vencedor"]
    print(f"vencedor (escolhido pela validação aninhada, antes de abrir o teste): {modelo} com {modo}\n")
    tabela = r["tabela_teste"].drop(columns="vencedor").set_index(["modo", "modelo"])
    print(tabela.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n{modelo} ({modo}) no teste, com intervalo de 95% (bootstrap):")
    linha = r["tabela_teste"].query("vencedor").iloc[0]
    for m in METRICAS_TESTE:
        inf, sup = r["ic"][m]
        print(f"  {m:<15} {linha[m]:.3f}  [{inf:.3f}; {sup:.3f}]")
    vn, fp, fn, vp = r["confusao"].ravel()
    print(f"\nmatriz de confusão: VP {vp} | FN {fn} | FP {fp} | VN {vn}")


def main():
    parser = argparse.ArgumentParser(description="Avaliação final no conjunto de teste (aberto uma única vez).")
    parser.add_argument("--resultados-dir", type=Path, default=Path("data/resultados"))
    parser.add_argument("--teste", type=Path, default=Path("data/teste.csv"))
    parser.add_argument("--grade", choices=GRADES, default="enxuta")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows usa cp1252 e quebra os acentos
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    r = avaliar(args.resultados_dir, args.teste, args.grade, args.n_bootstrap, args.seed)
    imprimir(r)
    gravar(r, args.resultados_dir, args.grade)
    print(f"\ngravado em {args.resultados_dir}: avaliacao_teste_{args.grade}.csv, vencedor_{args.grade}.json, "
          f"curvas ROC (roc_oof_*, roc_teste_*) e confusao_vencedor_{args.grade}.png")


if __name__ == "__main__":
    main()
