import pandas as pd
import pytest

from sepse import schema
from sepse.dados_sinteticos import gerar_dados
from sepse.io import ler_csv, salvar_csv


@pytest.fixture(scope="module")
def df():
    return gerar_dados(n_pacientes=5000, prevalencia=0.20, perc_faltantes=0.10, semente=1)


def test_mesma_semente_gera_mesmos_dados():
    pd.testing.assert_frame_equal(gerar_dados(300, semente=7), gerar_dados(300, semente=7))


def test_sementes_diferentes_geram_dados_diferentes():
    assert not gerar_dados(300, semente=1).equals(gerar_dados(300, semente=2))


def test_colunas_do_schema_na_ordem(df):
    assert tuple(df.columns) == schema.ORDEM_COLUNAS


def test_valores_dentro_das_faixas(df):
    for c in schema.COLUNAS:
        valores = df[c.nome].dropna()
        if c.tipo == schema.CATEGORICA:
            assert set(valores.unique()) <= set(c.valores), c.nome
        else:
            assert valores.min() >= c.minimo and valores.max() <= c.maximo, c.nome


def test_peso_cresce_com_idade_gestacional(df):
    medias = df.groupby("idade_gestacional_sem")["peso_nascer_g"].mean()
    assert medias.loc[28:30].mean() < medias.loc[33:35].mean() < medias.loc[38:40].mean()


@pytest.mark.parametrize("dispositivo", ["vmi", "cvc", "npt"])
def test_dias_zero_quando_nao_usou(df, dispositivo):
    dias = df.loc[df[dispositivo] == 0, f"{dispositivo}_dias"].dropna()
    assert (dias == 0).all()


def test_dias_ate_diagnostico_so_em_septicos(df):
    assert df.loc[df[schema.ALVO] == 0, "dias_ate_diagnostico"].isna().all()


def test_prevalencia_proxima_da_pedida(df):
    assert abs(df[schema.ALVO].mean() - 0.20) < 0.03


def test_taxa_de_faltantes(df):
    assert abs(df["peso_nascer_g"].isna().mean() - 0.10) < 0.02
    assert df["pct"].isna().mean() > 0.5
    assert df[schema.ID].notna().all() and df[schema.ALVO].notna().all()


def test_sinal_plantado(df):
    medias = df.groupby(schema.ALVO)[["pcr", "idade_gestacional_sem", "cvc", "plaquetas"]].mean()
    assert medias.loc[1, "pcr"] > medias.loc[0, "pcr"]
    assert medias.loc[1, "idade_gestacional_sem"] < medias.loc[0, "idade_gestacional_sem"]
    assert medias.loc[1, "cvc"] > medias.loc[0, "cvc"]
    assert medias.loc[1, "plaquetas"] < medias.loc[0, "plaquetas"]


def test_ida_e_volta_csv(tmp_path):
    original = gerar_dados(300, semente=3)
    caminho = tmp_path / "dados.csv"
    salvar_csv(original, caminho)
    pd.testing.assert_frame_equal(ler_csv(caminho), original)
