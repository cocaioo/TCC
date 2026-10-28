import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gerador_seed import EFEITO_DP, EXAMES, gerar_seed  # noqa: E402

SINAL = 0.6


@pytest.fixture(scope="module")
def df():
    return gerar_seed(n=3000, prevalencia=0.20, sinal=SINAL, seed=7)


def auc(x, y):
    """AUC por postos (Mann-Whitney), ignorando NaN."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    r = d["x"].rank()
    n1 = int((d["y"] == 1).sum())
    n0 = len(d) - n1
    return (r[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_esperada(nome, sinal):
    """Phi(d * sinal / raiz(2)): AUC de duas normais com a mesma variância separadas por d * sinal desvios."""
    return 0.5 * (1 + math.erf(EFEITO_DP[nome] * sinal / 2))


def test_colunas_dos_quadros_2_e_3(df):
    esperadas = [
        "sexo", "idade_gestacional_sem", "peso_nascer_g", "apgar_1min", "apgar_5min", "tipo_parto",
        "idade_materna", "raca_cor_rn", "raca_cor_mae", "n_gestacoes_anteriores", "itu_materna", "rpmo",
        "corioamnionite", "pre_eclampsia", "atb_materno_intraparto", "dias_utin", "vm_invasiva", "vm_dias",
        "cvc", "cvc_dias", "cateter_umbilical", "npt", "npt_dias", "atb_previo", "historia_sugestiva_infeccao",
        "criterios_disfuncao_organica", "ecn", "hiv_grau", "temp_min", "temp_max", "temp_media", "fc", "fr",
        "spo2", "pas", "pad", "vasopressor", "surfactante", "apneia_bradicardia_24h", "dias_ate_dx_sepse",
        "hemocultura_resultado", "hemocultura_agente", "perfil_sensibilidade", "hist_infeccao_ira",
        "colonizacao", "glasgow_neonatal", "news2", "qsofa", "id_paciente", "sepse_tardia",
    ] + [f"{e}_{s}" for e in EXAMES for s in ("pre", "pos24h")]
    assert set(esperadas) <= set(df.columns)
    assert not df.columns.duplicated().any()


def test_mesma_seed_gera_mesmos_dados():
    a = gerar_seed(n=200, seed=1)
    pd.testing.assert_frame_equal(a, gerar_seed(n=200, seed=1))
    assert not a.equals(gerar_seed(n=200, seed=2))


def test_prevalencia_exata_e_validacao_de_argumentos(df):
    assert df["sepse_tardia"].sum() == 600
    with pytest.raises(ValueError):
        gerar_seed(n=10, prevalencia=0)
    with pytest.raises(ValueError):
        gerar_seed(n=10, sinal=1.5)


def test_criterios_de_inclusao(df):
    assert (df["dias_utin"] > 3).all()
    assert (df["idade_gestacional_sem"] >= 22).all()
    assert (df["idade_gestacional_sem"] <= 41).all()


def test_dispositivos_coerentes_com_dias_de_internacao(df):
    for flag, dias in [("vm_invasiva", "vm_dias"), ("cvc", "cvc_dias"), ("npt", "npt_dias")]:
        assert (df[dias] <= df["dias_utin"]).all()
        assert (df.loc[df[flag] == 0, dias] == 0).all()
        assert (df.loc[df[flag] == 1, dias] >= 1).all()


def test_coerencias_do_parto_e_da_internacao(df):
    assert (df["apgar_5min"] >= df["apgar_1min"]).all()
    assert (df.loc[df["idade_gestacional_sem"] >= 32, "hiv_grau"] == 0).all()
    classes = [c for c in df.columns if c.startswith("atb_prev_")]
    assert (df.loc[df["atb_previo"] == 0, classes].sum(axis=1) == 0).all()
    assert (df.loc[df["atb_previo"] == 1, classes].sum(axis=1) >= 1).all()
    assert (df.loc[df["vasopressor"] == 1, "criterios_disfuncao_organica"] == 1).all()


def test_faixas_fisiologicas(df):
    assert df["temp_min"].le(df["temp_media"]).all() and df["temp_media"].le(df["temp_max"]).all()
    assert df["pad"].lt(df["pas"]).all()
    assert df["spo2"].between(70, 100).all()
    assert df["glasgow_neonatal"].between(3, 15).all()
    assert df["peso_nascer_g"].ge(400).all()
    assert df["ph_pre"].dropna().between(6.9, 7.55).all()


def test_relacoes_entre_exames(df):
    for s in ("pre", "pos24h"):
        completo = df[[f"bilirrubina_total_{s}", f"bilirrubina_direta_{s}", f"bilirrubina_indireta_{s}"]].dropna()
        soma = completo[f"bilirrubina_direta_{s}"] + completo[f"bilirrubina_indireta_{s}"]
        assert np.allclose(soma, completo[f"bilirrubina_total_{s}"], atol=0.11)
        neutro = df[[f"leucocitos_{s}", f"neutrofilos_seg_{s}", f"bastoes_{s}"]].dropna()
        assert (neutro[f"neutrofilos_seg_{s}"] + neutro[f"bastoes_{s}"] <= neutro[f"leucocitos_{s}"] + 1).all()
        assert df[f"relacao_it_{s}"].dropna().between(0, 1).all()


def test_escores_nos_intervalos(df):
    assert df["qsofa"].between(0, 3).all()
    assert df["news2"].between(0, 20).all()


def test_desfecho_aparece_so_em_dias_ate_dx(df):
    septico = df["sepse_tardia"] == 1
    assert df.loc[~septico, "dias_ate_dx_sepse"].isna().all()
    dx = df.loc[septico, "dias_ate_dx_sepse"]
    assert dx.notna().all() and dx.ge(3).all() and (dx <= df.loc[septico, "dias_utin"]).all()


@pytest.mark.parametrize("nome", ["pcr", "lactato", "plaquetas", "pct"])
def test_sinal_reproduz_efeito_esperado(df, nome):
    assert auc(df[f"{nome}_pre"], df["sepse_tardia"]) == pytest.approx(auc_esperada(nome, SINAL), abs=0.05)


def test_sem_sinal_exames_nao_separam_as_classes():
    sem_sinal = gerar_seed(n=3000, sinal=0, seed=7)
    for nome in ("pcr", "lactato", "plaquetas", "fc"):
        coluna = f"{nome}_pre" if nome != "fc" else "fc"
        assert auc(sem_sinal[coluna], sem_sinal["sepse_tardia"]) == pytest.approx(0.5, abs=0.05)


def test_fracao_de_ausentes_e_controlada_por_parametro():
    sem = gerar_seed(n=500, ausentes=0, seed=3)
    exames = [c for c in sem.columns if c.endswith(("_pre", "_pos24h"))]
    assert sem[exames].isna().sum().sum() == 0

    com = gerar_seed(n=3000, ausentes=1, seed=3)
    assert com["pct_pre"].isna().mean() == pytest.approx(0.45, abs=0.04)
    assert com["leucocitos_pre"].isna().mean() == pytest.approx(0.02, abs=0.015)
    assert com["pcr_pos24h"].isna().mean() > com["pcr_pre"].isna().mean()
    # variáveis demográficas e clínicas ficam completas
    assert com[["idade_gestacional_sem", "fc", "cvc", "dias_utin", "news2"]].isna().sum().sum() == 0


def test_microbiologia_coerente(df):
    positivas = df["hemocultura_resultado"] == "positiva"
    assert df.loc[positivas, "hemocultura_agente"].notna().all()
    assert df.loc[~positivas, ["hemocultura_agente", "perfil_sensibilidade"]].isna().all().all()
    # só uma parte dos sépticos tem hemocultura positiva (sepse clínica, A41.9)
    taxa = positivas[df["sepse_tardia"] == 1].mean()
    assert 0.15 < taxa < 0.45
