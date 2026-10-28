import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gerador_seed import gerar_seed  # noqa: E402
from preprocessamento import (  # noqa: E402
    COL_ALVO, COL_EXCLUIDA, COL_ID, auditar, carregar, dividir, separar_x_y,
)


@pytest.fixture(scope="module")
def df():
    return gerar_seed(n=1000, prevalencia=0.20, seed=11)


# --- Passo 1: carregar e auditar -------------------------------------------------------


def test_carregar_le_o_csv(tmp_path, df):
    caminho = tmp_path / "dados.csv"
    df.to_csv(caminho, index=False)
    lido = carregar(caminho)
    assert lido.shape == df.shape


def test_carregar_exige_id_e_alvo(tmp_path, df):
    caminho = tmp_path / "dados.csv"
    df.drop(columns=[COL_ALVO]).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="obrigatórias"):
        carregar(caminho)


def test_carregar_rejeita_alvo_nao_binario(tmp_path, df):
    caminho = tmp_path / "dados.csv"
    ruim = df.copy()
    ruim.loc[0, COL_ALVO] = 2
    ruim.to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="0 e 1"):
        carregar(caminho)


def test_auditoria_resume_os_dados(df):
    a = auditar(df)
    assert a["n_linhas"] == 1000
    assert a["n_positivos"] == 200
    assert a["n_preditoras"] == df.shape[1] - 2
    assert a["ids_duplicados"] == 0
    assert a["violacoes_ig"] == 0 and a["violacoes_dias_utin"] == 0
    assert a["taxa_ausente"].is_monotonic_decreasing
    assert a["taxa_ausente"]["pct_pre"] > 0.3


def test_auditoria_sinaliza_vazamento_estrutural(df):
    a = auditar(df)
    assert COL_EXCLUIDA in a["suspeitas_vazamento"]
    # variáveis legítimas, mesmo com AUC alta, não são suspeitas
    assert "pcr_pre" not in a["suspeitas_vazamento"]


def test_auditoria_acha_violacao_dos_criterios_de_inclusao(df):
    ruim = df.copy()
    ruim.loc[0, "idade_gestacional_sem"] = 20
    ruim.loc[1, "dias_utin"] = 2
    ruim.loc[2, COL_ID] = ruim.loc[3, COL_ID]
    a = auditar(ruim)
    assert a["violacoes_ig"] == 1
    assert a["violacoes_dias_utin"] == 1
    assert a["ids_duplicados"] == 1


def test_auditoria_acha_colunas_quase_constantes(df):
    ruim = df.copy()
    ruim["constante"] = 1
    assert "constante" in auditar(ruim)["quase_constantes"]


# --- Passo 2: dividir -----------------------------------------------------------------


def test_divisao_80_20_estratificada(df):
    treino, teste = dividir(df, teste=0.20, seed=42)
    assert len(treino) == 800 and len(teste) == 200
    assert treino[COL_ALVO].sum() == 160 and teste[COL_ALVO].sum() == 40


def test_divisao_nao_repete_nem_perde_pacientes(df):
    treino, teste = dividir(df, seed=42)
    assert set(treino[COL_ID]).isdisjoint(teste[COL_ID])
    assert set(treino[COL_ID]) | set(teste[COL_ID]) == set(df[COL_ID])


def test_divisao_reprodutivel_e_depende_da_semente(df):
    a, _ = dividir(df, seed=1)
    b, _ = dividir(df, seed=1)
    c, _ = dividir(df, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert set(a[COL_ID]) != set(c[COL_ID])


def test_separar_x_y_tira_id_alvo_e_vazamento_estrutural(df):
    X, y = separar_x_y(df)
    assert not {COL_ID, COL_ALVO, COL_EXCLUIDA} & set(X.columns)
    assert X.shape[1] == df.shape[1] - 3
    assert y.equals(df[COL_ALVO])
