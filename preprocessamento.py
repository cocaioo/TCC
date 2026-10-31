"""Pré-processamento dos dados de sepse neonatal tardia (etapas 2 e 3 do KDD).

Funciona com o CSV do seed (`gerador_seed.py`) e, no futuro, com a extração real do SOUL MV; por isso não importa
nada do gerador: os grupos de colunas abaixo são constantes próprias.

Passo 1: carregar e auditar (nada aqui "aprende" dos dados nem altera o DataFrame).
Passo 2: dividir em treino e teste, de forma estratificada, e gravar os dois conjuntos.
Passo 3: pré-processador (hemocultura, log1p, imputação pela mediana, indicador de ausência, escala, one-hot).
         Ele é devolvido SEM ajuste: só o treino (ou cada dobra da validação cruzada) o ajusta, para não vazar.

Uso:
    python preprocessamento.py --entrada data/seed_sepse.csv --seed 42
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------------------
# Grupos de colunas (ajustar quando o conector real existir)
# --------------------------------------------------------------------------------------

COL_ID = "id_paciente"
COL_ALVO = "sepse_tardia"

# Vazamento estrutural: só existe para quem teve sepse. Sai sempre das variáveis de entrada.
COL_EXCLUIDA = "dias_ate_dx_sepse"

# Preditoras previstas no PDF cujo valor pode ser CONSEQUÊNCIA da sepse (totais da internação e resultado da
# hemocultura). Ficam por padrão; a comparação com e sem elas saiu do escopo até a definição da pergunta do modelo (plano, S5).
COLS_SINALIZADAS = [
    "dias_utin", "vm_dias", "cvc_dias", "npt_dias",
    "hemocultura_resultado", "hemocultura_agente", "perfil_sensibilidade",
]

COLS_OBRIGATORIAS = [COL_ID, COL_ALVO]

# Critérios de inclusão do projeto (p. 37): IG >= 22 semanas e internação > 72 h (> 3 dias).
IG_MINIMA_SEMANAS = 22
DIAS_UTIN_MINIMO = 3

# Limiares da auditoria
LIMIAR_QUASE_CONSTANTE = 0.95  # fração mínima do valor mais frequente
LIMIAR_AUC_VAZAMENTO = 0.95  # AUC de uma coluna sozinha acima disso é suspeita de vazamento

# Transformação (passo 3)
COLS_CATEGORICAS = [
    "sexo", "tipo_parto", "raca_cor_rn", "raca_cor_mae", "colonizacao",
    "hemocultura_resultado", "hemocultura_agente", "perfil_sensibilidade",
]
SUFIXOS_EXAME = ("_pre", "_pos24h")  # janela da última medida antes da suspeita e das primeiras até 24 h depois

# Exames de cauda longa (poucos valores muito altos): recebem log1p antes da padronização. Lista por conhecimento
# clínico, e não calculada dos dados, para valer igual com o dado real. Ficam de fora os simétricos (glicemia,
# hemoglobina, hematócrito, pH, Na, K, Ca ionizado) e a relação IT, que já é limitada a 0-1.
BASES_LOG1P = [
    "leucocitos", "neutrofilos_seg", "bastoes", "plaquetas", "pcr", "pct", "creatinina",
    "bilirrubina_total", "bilirrubina_direta", "bilirrubina_indireta", "lactato", "tgo", "tgp",
    "relacao_neutrofilos_linfocitos",
]

# Só exames com pelo menos esta fração de ausentes (no treino) ganham a coluna "foi medido?".
LIMIAR_INDICADOR_PADRAO = 0.20


# --------------------------------------------------------------------------------------
# Carregar
# --------------------------------------------------------------------------------------


def carregar(caminho):
    """Lê o CSV e confere se as colunas obrigatórias (id e alvo) existem e se o alvo é 0/1."""
    df = pd.read_csv(caminho)
    faltando = [c for c in COLS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas obrigatórias ausentes em {caminho}: {faltando}")
    if not set(df[COL_ALVO].dropna().unique()) <= {0, 1} or df[COL_ALVO].isna().any():
        raise ValueError(f"'{COL_ALVO}' deve conter apenas 0 e 1, sem valores ausentes")
    return df


# --------------------------------------------------------------------------------------
# Auditar
# --------------------------------------------------------------------------------------


def _auc(valores, alvo):
    """AUC por postos (Mann-Whitney) de `valores` contra `alvo`; ignora NaN; devolve max(AUC, 1 - AUC)."""
    d = pd.DataFrame({"x": valores, "y": alvo}).dropna()
    n1 = int((d["y"] == 1).sum())
    n0 = len(d) - n1
    if n1 == 0 or n0 == 0 or d["x"].nunique() < 2:
        return np.nan
    auc = (d["x"].rank()[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(auc, 1 - auc)


def auditar(df):
    """Resumo de qualidade dos dados. Só lê; não altera nada.

    Vazamento: para cada coluna numérica calcula a AUC do valor sozinho e, se houver NaN, a AUC do indicador
    "está ausente". Uma coluna que prevê o alvo quase perfeitamente (ex.: NaN se e somente se não houve sepse)
    é suspeita de vazamento de alvo.
    """
    y = df[COL_ALVO]
    preditoras = df.drop(columns=[COL_ID, COL_ALVO])
    numericas = preditoras.select_dtypes("number")

    taxa_ausente = preditoras.isna().mean()
    taxa_ausente = taxa_ausente[taxa_ausente > 0].sort_values(ascending=False)

    quase_constantes = {}
    for coluna in preditoras.columns:
        frequencias = preditoras[coluna].value_counts(normalize=True, dropna=True)
        if len(frequencias) and frequencias.iloc[0] >= LIMIAR_QUASE_CONSTANTE:
            quase_constantes[coluna] = float(frequencias.iloc[0])

    auc_por_coluna = {}
    for coluna in numericas.columns:
        candidatas = [_auc(numericas[coluna], y)]
        if numericas[coluna].isna().any():
            candidatas.append(_auc(numericas[coluna].isna().astype(int), y))
        auc_por_coluna[coluna] = float(np.nanmax(candidatas)) if not np.isnan(candidatas).all() else np.nan
    auc_por_coluna = pd.Series(auc_por_coluna).sort_values(ascending=False)

    return {
        "n_linhas": len(df),
        "n_preditoras": preditoras.shape[1],
        "n_positivos": int(y.sum()),
        "prevalencia": float(y.mean()),
        "ids_duplicados": int(df[COL_ID].duplicated().sum()),
        "violacoes_ig": int((df["idade_gestacional_sem"] < IG_MINIMA_SEMANAS).sum())
        if "idade_gestacional_sem" in df else None,
        "violacoes_dias_utin": int((df["dias_utin"] <= DIAS_UTIN_MINIMO).sum()) if "dias_utin" in df else None,
        "taxa_ausente": taxa_ausente,
        "quase_constantes": quase_constantes,
        "auc_por_coluna": auc_por_coluna,
        "suspeitas_vazamento": auc_por_coluna[auc_por_coluna > LIMIAR_AUC_VAZAMENTO].index.tolist(),
    }


def imprimir_auditoria(a, top=10):
    print(f"Linhas: {a['n_linhas']}  |  preditoras: {a['n_preditoras']}  |  "
          f"sepse tardia: {a['n_positivos']} ({a['prevalencia']:.1%})")
    print(f"ids duplicados: {a['ids_duplicados']}  |  IG < {IG_MINIMA_SEMANAS} sem: {a['violacoes_ig']}  |  "
          f"dias_utin <= {DIAS_UTIN_MINIMO}: {a['violacoes_dias_utin']}")

    ausentes = a["taxa_ausente"]
    print(f"\nColunas com ausentes: {len(ausentes)} (maiores {min(top, len(ausentes))}):")
    for coluna, taxa in ausentes.head(top).items():
        print(f"  {coluna:<42} {taxa:6.1%}")

    print(f"\nQuase constantes (>= {LIMIAR_QUASE_CONSTANTE:.0%} no mesmo valor): {len(a['quase_constantes'])}")
    for coluna, fracao in sorted(a["quase_constantes"].items(), key=lambda kv: -kv[1]):
        print(f"  {coluna:<42} {fracao:6.1%}")

    print(f"\nAUC de uma coluna sozinha contra o alvo (maiores {top}):")
    for coluna, auc in a["auc_por_coluna"].head(top).items():
        print(f"  {coluna:<42} {auc:6.3f}")
    if a["suspeitas_vazamento"]:
        print(f"\nATENÇÃO, suspeitas de vazamento de alvo (AUC > {LIMIAR_AUC_VAZAMENTO}): "
              f"{a['suspeitas_vazamento']}")
    else:
        print("\nNenhuma suspeita de vazamento por AUC isolada.")


# --------------------------------------------------------------------------------------
# Dividir
# --------------------------------------------------------------------------------------


def dividir(df, teste=0.20, seed=42):
    """Separa em treino e teste (80/20, p. 45), estratificado pelo alvo: as duas partes mantêm a prevalência.

    Devolve DataFrames completos (com `id_paciente` e alvo), para a divisão poder ser gravada e auditada.
    O conjunto de teste só deve ser aberto uma vez, no fim, para a avaliação final.
    """
    treino, teste_df = train_test_split(df, test_size=teste, stratify=df[COL_ALVO], random_state=seed)
    return treino.reset_index(drop=True), teste_df.reset_index(drop=True)


def separar_x_y(df):
    """X = variáveis de entrada (sem id, alvo e a coluna de vazamento estrutural); y = alvo."""
    return df.drop(columns=[COL_ID, COL_ALVO, COL_EXCLUIDA], errors="ignore"), df[COL_ALVO]


# --------------------------------------------------------------------------------------
# Transformar
# --------------------------------------------------------------------------------------


def tratar_ausentes_estruturais(X):
    """Hemocultura: o vazio de `agente` e `perfil_sensibilidade` quase sempre significa "não se aplica".

    Cultura negativa não tem agente; isso não é dado perdido e não pode ser imputado com mediana ou moda.
    Função pura: não aprende nada dos dados, então pode rodar antes da divisão ou dentro de cada dobra.
    """
    X = X.copy()
    resultado = X["hemocultura_resultado"]
    preenchimento = np.select(
        [resultado == "negativa", resultado == "positiva"], ["nao_se_aplica", "nao_informado"],
        default="nao_realizada")
    for coluna in ("hemocultura_agente", "perfil_sensibilidade"):
        X[coluna] = X[coluna].where(X[coluna].notna(), preenchimento)
    X["hemocultura_resultado"] = resultado.fillna("nao_realizada")
    return X


def aplicar_log1p(X):
    """`log1p` nos exames de cauda longa (BASES_LOG1P, nas duas janelas). Função pura; NaN passa intacto."""
    colunas = [f"{base}{sufixo}" for base in BASES_LOG1P for sufixo in SUFIXOS_EXAME
               if f"{base}{sufixo}" in X.columns]
    if (X[colunas] <= -1).any().any():
        raise ValueError("valor <= -1 em exame com log1p (ex.: código de ausente do sistema, como -999)")
    X = X.copy()
    X[colunas] = np.log1p(X[colunas])
    return X


def _para_float(X):
    """Matriz final com um único tipo (float): os indicadores saem como bool e as binárias como int."""
    return X.astype(float)


def _selecionar_binarias(X, categoricas, exames):
    """Colunas 0/1 (valores observados dentro de {0, 1}); passam sem escala. Roda no `fit`, com os dados recebidos."""
    return [c for c in X.columns
            if c not in categoricas and c not in exames and (v := set(X[c].dropna().unique())) and v <= {0, 1}]


def _selecionar_numericas(X, categoricas, exames):
    """As demais colunas (numéricas e exames): mediana -> padronização. Roda no `fit`."""
    binarias = _selecionar_binarias(X, categoricas, exames)
    return [c for c in X.columns if c not in categoricas and c not in binarias]


def _selecionar_com_indicador(X, exames, limiar):
    """Exames com ausência >= `limiar` nos dados recebidos no `fit` ganham a coluna "foi medido?"."""
    return [c for c in exames if X[c].isna().mean() >= limiar]


def construir_preprocessador(X_treino, limiar_indicador=LIMIAR_INDICADOR_PADRAO):
    """Devolve o pré-processador (`Pipeline` do scikit-learn) SEM ajuste; saída em DataFrame com nomes de colunas.

    Etapas: hemocultura -> log1p -> ColumnTransformer com
      binárias (0/1)       passam sem escala;
      numéricas e exames   mediana do treino -> padronização;
      indicador de ausência ("foi medido?") só para exames com ausência >= `limiar_indicador` no treino;
      categóricas          ausente vira "ausente" -> one-hot (categoria nova no teste vira zeros).

    `X_treino` só fornece os NOMES das colunas (categóricas e exames, que se reconhecem pelo nome). O que depende dos
    VALORES (quais colunas são binárias; quais exames passam do limiar de ausência) é decidido no `fit`, com os dados
    que o pipeline recebe: o treino, ou só as dobras de estudo na validação cruzada. Assim, a dobra de validação não
    influencia nem a estrutura do modelo. O alvo nunca é usado.
    """
    categoricas = [c for c in COLS_CATEGORICAS if c in X_treino.columns]
    exames = [c for c in X_treino.columns if c.endswith(SUFIXOS_EXAME)]

    colunas = ColumnTransformer([
        ("bin", "passthrough", partial(_selecionar_binarias, categoricas=categoricas, exames=exames)),
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
         partial(_selecionar_numericas, categoricas=categoricas, exames=exames)),
        ("ind", MissingIndicator(features="all"),
         partial(_selecionar_com_indicador, exames=exames, limiar=limiar_indicador)),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="ausente")),
                          ("codificar", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                                      sparse_output=False))]), categoricas),
    ], verbose_feature_names_out=False)

    return Pipeline([
        ("estrutural", FunctionTransformer(tratar_ausentes_estruturais)),
        ("log1p", FunctionTransformer(aplicar_log1p)),
        ("colunas", colunas),
        ("float", FunctionTransformer(_para_float)),
    ]).set_output(transform="pandas")


def main():
    parser = argparse.ArgumentParser(description="Auditoria e pré-processamento dos dados de sepse neonatal.")
    parser.add_argument("--entrada", type=Path, default=Path("data/seed_sepse.csv"))
    parser.add_argument("--saida-dir", type=Path, default=Path("data"), help="onde gravar treino.csv e teste.csv")
    parser.add_argument("--teste", type=float, default=0.20, help="fração do conjunto de teste (padrão 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="semente da divisão treino/teste")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows usa cp1252 e quebra os acentos
    df = carregar(args.entrada)
    imprimir_auditoria(auditar(df))

    treino, teste = dividir(df, args.teste, args.seed)
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    treino.to_csv(args.saida_dir / "treino.csv", index=False, encoding="utf-8")
    teste.to_csv(args.saida_dir / "teste.csv", index=False, encoding="utf-8")
    print("\nDivisão estratificada (semente %d):" % args.seed)
    for nome, parte in (("treino", treino), ("teste", teste)):
        print(f"  {nome:<7} {len(parte):>5} linhas | sepse tardia: {int(parte[COL_ALVO].sum()):>3} "
              f"({parte[COL_ALVO].mean():.1%})")
    X, _ = separar_x_y(treino)
    print(f"  variáveis de entrada (X): {X.shape[1]}")

    # O teste NÃO é transformado aqui: só é aberto na avaliação final.
    matriz = construir_preprocessador(X).fit_transform(X)
    indicadores = int(matriz.columns.str.startswith("missingindicator_").sum())
    print(f"  matriz do treino após o pré-processador: {matriz.shape[0]} x {matriz.shape[1]} "
          f"({indicadores} indicadores de ausência), {int(matriz.isna().sum().sum())} ausentes")


if __name__ == "__main__":
    main()
