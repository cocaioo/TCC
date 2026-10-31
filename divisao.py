"""Divisão dos dados em treino e teste (etapa genérica: não trata nem transforma os dados).

Recebe uma tabela com uma linha por recém-nascido, uma coluna de identificação e o desfecho 0/1. Os nomes padrão das
duas colunas vêm do dicionário de variáveis do projeto (Apêndice A): `id_prontuario` e `sepse_tardia`.

O tratamento dos dados (limpeza, ausentes, codificação, escolha das variáveis) NÃO está aqui: depende dos dados reais e
de definições ainda em aberto. Ele entrará como o passo de pré-processamento do pipeline (`modelos.construir_pipeline`).

Uso:
    python divisao.py --entrada dados.csv --saida-dir pasta --seed 42
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

COL_ID = "id_prontuario"
COL_ALVO = "sepse_tardia"


def carregar(caminho):
    """Lê o CSV e confere se as colunas de identificação e de desfecho existem e se o desfecho é 0/1, sem ausentes."""
    df = pd.read_csv(caminho)
    faltando = [c for c in (COL_ID, COL_ALVO) if c not in df.columns]
    if faltando:
        raise ValueError(f"colunas obrigatórias ausentes em {caminho}: {faltando}")
    if df[COL_ALVO].isna().any() or not set(df[COL_ALVO].unique()) <= {0, 1}:
        raise ValueError(f"'{COL_ALVO}' deve conter apenas 0 e 1, sem valores ausentes")
    return df


def dividir(df, teste=0.20, seed=42):
    """Separa em treino e teste (80/20, p. 45 do projeto), estratificado pelo desfecho: as duas partes mantêm a mesma
    proporção de sepse. O conjunto de teste só deve ser aberto uma vez, na avaliação final."""
    treino, teste_df = train_test_split(df, test_size=teste, stratify=df[COL_ALVO], random_state=seed)
    return treino.reset_index(drop=True), teste_df.reset_index(drop=True)


def separar_x_y(df, excluir=()):
    """X = variáveis de entrada (sem o identificador, o desfecho e as colunas em `excluir`); y = desfecho."""
    return df.drop(columns=[COL_ID, COL_ALVO, *excluir], errors="ignore"), df[COL_ALVO]


def main():
    parser = argparse.ArgumentParser(description="Divisão estratificada em treino e teste.")
    parser.add_argument("--entrada", type=Path, required=True, help="CSV com uma linha por recém-nascido")
    parser.add_argument("--saida-dir", type=Path, default=Path("data"), help="onde gravar treino.csv e teste.csv")
    parser.add_argument("--teste", type=float, default=0.20, help="fração do conjunto de teste (padrão 0.20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows usa cp1252 e quebra os acentos
    treino, teste = dividir(carregar(args.entrada), args.teste, args.seed)
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    treino.to_csv(args.saida_dir / "treino.csv", index=False, encoding="utf-8")
    teste.to_csv(args.saida_dir / "teste.csv", index=False, encoding="utf-8")
    for nome, parte in (("treino", treino), ("teste", teste)):
        print(f"{nome:<7} {len(parte):>5} linhas | {COL_ALVO}: {int(parte[COL_ALVO].sum())} "
              f"({parte[COL_ALVO].mean():.1%})")


if __name__ == "__main__":
    main()
