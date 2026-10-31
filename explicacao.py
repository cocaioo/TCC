"""Explicação do melhor modelo (SHAP) e modelo final salvo (etapa 5 do KDD: interpretação).

Parte do vencedor escolhido por `avaliacao.py` (`vencedor_<grade>.json`), que já foi medido no teste. Nada aqui muda o
modelo nem a escolha: o SHAP só descreve o que o modelo faz.

SHAP: para cada paciente, a probabilidade prevista é decomposta em "valor-base" (a probabilidade média do modelo nos
pacientes de referência) mais uma contribuição por variável; as contribuições somam exatamente a diferença. Usa-se o
explicador por permutação na escala de probabilidade, o mesmo para os 8 algoritmos, sobre as variáveis já
pré-processadas (as que o modelo enxerga). Referência: até 100 pacientes do treino; explicados: os pacientes do teste.

Modelo final: `modelo_final.joblib` (o pipeline inteiro, do CSV bruto à probabilidade) e `modelo_final.json` (o que é,
como foi escolhido, quais colunas espera, limiar e métricas). `prever` mostra como outro programa o usaria.

Uso:
    python explicacao.py --grade enxuta
"""

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.pipeline import Pipeline as PipelineSk
from sklearn.preprocessing import FunctionTransformer

from avaliacao import SUPERFICIE, TINTA, carregar_modelo
from modelos import GRADES, catalogo
from divisao import COL_ID, carregar, separar_x_y
from validacao import _metricas

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_REFERENCIA = 100
N_VARIAVEIS_GRAFICO = 15


# --------------------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------------------


def separar_pipeline(pipeline):
    """`(pré-processador ajustado, modelo)`. O passo `smote` fica de fora: ele só age no treino, nunca na previsão.
    Sem pré-processamento no pipeline, o primeiro elemento é a identidade (devolve a tabela como chega)."""
    passos = [(nome, passo) for nome, passo in pipeline.steps if nome not in ("smote", "modelo")]
    return (PipelineSk(passos) if passos else FunctionTransformer()), pipeline["modelo"]


def _transformar(prep, X):
    """Aplica o pré-processamento e devolve um DataFrame com nomes de colunas (os nomes aparecem no SHAP).

    Se o pré-processamento não informar os nomes, as colunas viram `x0, x1, ...`."""
    saida = prep.transform(X)
    if isinstance(saida, pd.DataFrame):
        return saida
    try:
        nomes = prep.get_feature_names_out()
    except AttributeError:  # algum passo não informa os nomes das colunas que devolve
        nomes = None
    if nomes is None or len(nomes) != saida.shape[1]:
        nomes = [f"x{i}" for i in range(saida.shape[1])]
    return pd.DataFrame(saida, columns=list(map(str, nomes)), index=X.index)


def explicar(pipeline, X_referencia, X_explicar, seed=42):
    """Valores SHAP (objeto `shap.Explanation`) da probabilidade de sepse, nas variáveis pré-processadas.

    Propriedade garantida pelo método: `base_values[i] + values[i].sum() == probabilidade prevista do paciente i`.
    """
    prep, modelo = separar_pipeline(pipeline)
    referencia = _transformar(prep, X_referencia)
    if len(referencia) > N_REFERENCIA:
        referencia = referencia.sample(N_REFERENCIA, random_state=seed)
    explicador = shap.explainers.Permutation(lambda M: modelo.predict_proba(M)[:, 1],
                                             shap.maskers.Independent(referencia), seed=seed)
    # O método exige pelo menos 2 x variáveis + 1 avaliações por paciente; o padrão (500) não basta acima de 249.
    return explicador(_transformar(prep, X_explicar), max_evals=max(500, 2 * referencia.shape[1] + 1), silent=True)


def importancia_global(explicacao):
    """Média do valor SHAP absoluto por variável (em pontos de probabilidade), da maior para a menor."""
    tabela = pd.DataFrame({"variavel": explicacao.feature_names,
                           "shap_medio_absoluto": np.abs(explicacao.values).mean(axis=0),
                           "shap_medio": explicacao.values.mean(axis=0)})
    return tabela.sort_values("shap_medio_absoluto", ascending=False, ignore_index=True)


def escolher_caso(probabilidades, limiar=0.5):
    """Índice do paciente usado como exemplo: o de probabilidade mediana entre os classificados como sepse (um alerta
    típico, nem o extremo de probabilidade 1). Sem nenhum alerta, o de maior probabilidade."""
    probabilidades = np.asarray(probabilidades)
    alertas = np.flatnonzero(probabilidades > limiar)
    if len(alertas) == 0:
        return int(np.argmax(probabilidades))
    ordenados = alertas[np.argsort(probabilidades[alertas], kind="stable")]
    return int(ordenados[(len(ordenados) - 1) // 2])


def figura_resumo(explicacao, titulo, caminho):
    """Beeswarm: cada ponto é um paciente; posição = contribuição para a probabilidade; cor = valor da variável."""
    shap.plots.beeswarm(explicacao, max_display=N_VARIAVEIS_GRAFICO, show=False, color=plt.get_cmap("coolwarm"))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Contribuição para a probabilidade de sepse (valor SHAP)")
    rotulos = [t.get_text().replace("Sum of", "Soma das outras").replace("other features", "variáveis")
               for t in fig.axes[0].get_yticklabels()]
    fig.axes[0].set_yticks(fig.axes[0].get_yticks(), rotulos)
    barra = fig.axes[-1]  # barra de cores do beeswarm
    barra.set_ylabel("Valor da variável")
    barra.set_yticks(barra.get_yticks(), ["Baixo", "Alto"][:len(barra.get_yticks())])
    _finalizar(titulo, caminho)


def figura_caso(explicacao, indice, titulo, caminho):
    """Waterfall de um paciente: do valor-base até a probabilidade prevista, variável por variável."""
    shap.plots.waterfall(explicacao[indice], max_display=N_VARIAVEIS_GRAFICO, show=False)
    fig = plt.gcf()
    eixo = fig.axes[0]
    eixo.set_yticks(eixo.get_yticks(), [t.get_text().replace("other features", "outras variáveis")
                                        for t in eixo.get_yticklabels()])
    fig.text(0.02, 0.01, "Valores à esquerda: variáveis como o modelo as recebe, depois do pré-processamento.",
             fontsize=8, color=TINTA)
    _finalizar(titulo, caminho, margem_direita=0.9)


def _finalizar(titulo, caminho, margem_direita=None):
    fig = plt.gcf()
    fig.set_size_inches(8, 6.5)
    fig.patch.set_facecolor(SUPERFICIE)
    fig.suptitle(titulo, color=TINTA, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if margem_direita:
        fig.subplots_adjust(right=margem_direita)
    fig.savefig(caminho, dpi=150, facecolor=SUPERFICIE)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Modelo final
# --------------------------------------------------------------------------------------


def _valor_json(valor):
    """Hiperparâmetro em forma gravável em JSON (tuplas viram listas; o resto, texto)."""
    if isinstance(valor, (bool, int, float, str)) or valor is None:
        return valor
    if isinstance(valor, (tuple, list)):
        return [_valor_json(v) for v in valor]
    return str(valor)


def salvar_modelo_final(pipeline, resumo_vencedor, colunas, resultados_dir):
    """Grava `modelo_final.joblib` e `modelo_final.json`; devolve o dicionário do JSON."""
    resultados_dir = Path(resultados_dir)
    grade = catalogo(resumo_vencedor["grade"])[resumo_vencedor["modelo"]][1]
    parametros = pipeline.get_params()
    metadados = {
        "algoritmo": resumo_vencedor["modelo"],
        "desbalanceamento": resumo_vencedor["modo"],
        "grade": resumo_vencedor["grade"],
        "hiperparametros": {nome.removeprefix("modelo__"): _valor_json(parametros[nome]) for nome in grade},
        "escolha": "maior F1 médio da validação cruzada aninhada (5 x 5) no treino, desempate pela AUROC",
        "limiar": resumo_vencedor["limiar"],
        "colunas_entrada": list(colunas),
        "metricas_validacao_aninhada": resumo_vencedor["validacao_aninhada"],
        "metricas_teste": resumo_vencedor["teste"],
        "metricas_teste_ic95": resumo_vencedor["teste_ic95"],
        "data": date.today().isoformat(),
        "versoes": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__},
    }
    joblib.dump(pipeline, resultados_dir / "modelo_final.joblib")
    (resultados_dir / "modelo_final.json").write_text(json.dumps(metadados, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    return metadados


def prever(df, resultados_dir=Path("data/resultados")):
    """Probabilidade de sepse tardia de cada paciente de `df` (uma linha por recém-nascido), com o modelo final.

    Confere as colunas contra `modelo_final.json`: faltar coluna é erro; coluna a mais é ignorada com aviso. Devolve uma
    Series indexada pela coluna de identificação (se existir) ou pelo índice de `df`.
    """
    resultados_dir = Path(resultados_dir)
    metadados = json.loads((resultados_dir / "modelo_final.json").read_text(encoding="utf-8"))
    esperadas = metadados["colunas_entrada"]
    faltando = [c for c in esperadas if c not in df.columns]
    if faltando:
        raise ValueError(f"faltam {len(faltando)} colunas de entrada esperadas pelo modelo: {faltando}")
    extras = [c for c in df.columns if c not in esperadas and c != COL_ID]
    if extras:
        warnings.warn(f"colunas ignoradas (o modelo não as usa): {extras}", UserWarning, stacklevel=2)
    pipeline = joblib.load(resultados_dir / "modelo_final.joblib")
    proba = pipeline.predict_proba(df[esperadas])[:, 1]
    indice = df[COL_ID] if COL_ID in df.columns else df.index
    return pd.Series(proba, index=pd.Index(indice, name=COL_ID), name="probabilidade_sepse")


# --------------------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------------------


def conferir_modelo(pipeline, X_teste, y_teste, resumo):
    """Garante que o modelo carregado é o que `avaliacao.py` mediu: as métricas do teste têm de bater com o resumo.

    Pega o caso de `validacao.py` rodado de novo (outra semente, outros dados) sem rodar `avaliacao.py` depois: o
    modelo salvo mudou, e o modelo final sairia com métricas de outro modelo."""
    medidas = _metricas(y_teste, pipeline.predict(X_teste), pipeline.predict_proba(X_teste)[:, 1])
    diferentes = [m for m, v in resumo["teste"].items() if not np.isclose(medidas[m], v, atol=1e-9)]
    if diferentes:
        raise ValueError(f"o modelo salvo não é o avaliado (métricas diferentes: {diferentes}); rode avaliacao.py de "
                         "novo depois de validacao.py")


def executar(resultados_dir, caminho_treino, caminho_teste, grade="enxuta", seed=42):
    """Explica o vencedor no teste, grava importância, figuras e o modelo final. Devolve um dicionário de resultados."""
    resultados_dir = Path(resultados_dir)
    caminho_resumo = resultados_dir / f"vencedor_{grade}.json"
    if not caminho_resumo.exists():
        raise FileNotFoundError(f"{caminho_resumo} não existe: rode avaliacao.py antes")
    resumo = json.loads(caminho_resumo.read_text(encoding="utf-8"))
    pipeline = carregar_modelo(resultados_dir, grade, resumo["modo"], resumo["modelo"])
    X_treino, _ = separar_x_y(carregar(caminho_treino))
    X_teste, y_teste = separar_x_y(carregar(caminho_teste))
    conferir_modelo(pipeline, X_teste, y_teste, resumo)

    explicacao = explicar(pipeline, X_treino, X_teste, seed)
    importancia = importancia_global(explicacao)
    importancia.to_csv(resultados_dir / f"importancia_shap_{grade}.csv", index=False, encoding="utf-8")
    nome = f"{resumo['modelo']} ({resumo['modo']})"
    figura_resumo(explicacao, f"SHAP no teste · {nome}", resultados_dir / f"shap_resumo_{grade}.png")
    caso = escolher_caso(explicacao.base_values + explicacao.values.sum(axis=1), resumo["limiar"])
    figura_caso(explicacao, caso, f"Explicação de um alerta (probabilidade mediana entre os alertas) · {nome}",
                resultados_dir / f"shap_caso_{grade}.png")

    metadados = salvar_modelo_final(pipeline, resumo, X_treino.columns, resultados_dir)
    return {"resumo": resumo, "explicacao": explicacao, "importancia": importancia, "metadados": metadados,
            "caso": caso}


def main():
    parser = argparse.ArgumentParser(description="SHAP do melhor modelo e modelo final salvo.")
    parser.add_argument("--resultados-dir", type=Path, default=Path("data/resultados"))
    parser.add_argument("--treino", type=Path, default=Path("data/treino.csv"))
    parser.add_argument("--teste", type=Path, default=Path("data/teste.csv"))
    parser.add_argument("--grade", choices=GRADES, default="enxuta")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows usa cp1252 e quebra os acentos
    r = executar(args.resultados_dir, args.treino, args.teste, args.grade, args.seed)
    print(f"modelo explicado: {r['resumo']['modelo']} ({r['resumo']['modo']})\n")
    print(f"variáveis de maior importância (média do |SHAP|, em pontos de probabilidade), top {N_VARIAVEIS_GRAFICO}:")
    for _, linha in r["importancia"].head(N_VARIAVEIS_GRAFICO).iterrows():
        print(f"  {linha['variavel']:<42} {linha['shap_medio_absoluto']:.4f}")
    print(f"\ngravado em {args.resultados_dir}: importancia_shap_{args.grade}.csv, shap_resumo_{args.grade}.png, "
          f"shap_caso_{args.grade}.png, modelo_final.joblib e modelo_final.json")


if __name__ == "__main__":
    main()
