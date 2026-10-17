"""Leitura e escrita de CSV aplicando os tipos do schema."""

from pathlib import Path

import pandas as pd

from sepse import schema


def _dtype(coluna: schema.Coluna):
    if coluna.tipo in (schema.BINARIA, schema.ORDINAL):
        return "Int64"  # inteiro que aceita faltante
    if coluna.tipo == schema.CATEGORICA:
        return pd.CategoricalDtype(list(coluna.valores))
    return "float64"


def aplicar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Converte cada coluna para o tipo definido no schema e ordena as colunas."""
    tipos = {c.nome: _dtype(c) for c in schema.COLUNAS if c.nome in df.columns}
    tipos[schema.ID] = "int64"
    if schema.ALVO in df.columns:
        tipos[schema.ALVO] = "int64"
    df = df.astype(tipos)
    return df[[c for c in schema.ORDEM_COLUNAS if c in df.columns]]


def salvar_csv(df: pd.DataFrame, caminho: str | Path) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")


def ler_csv(caminho: str | Path) -> pd.DataFrame:
    return aplicar_tipos(pd.read_csv(caminho, encoding="utf-8"))
