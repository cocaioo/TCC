import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dados_genericos import tabela_generica  # noqa: E402
from divisao import COL_ALVO, COL_ID, carregar, dividir, separar_x_y  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return tabela_generica(n=1000, prevalencia=0.20, seed=11)


def test_carregar_le_o_csv(tmp_path, df):
    caminho = tmp_path / "dados.csv"
    df.to_csv(caminho, index=False)
    assert carregar(caminho).shape == df.shape


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


def test_divisao_80_20_estratificada(df):
    treino, teste = dividir(df, teste=0.20, seed=42)
    assert len(treino) == 800 and len(teste) == 200
    # a proporção do desfecho é a mesma nas duas partes (difere no máximo pelo arredondamento de um paciente)
    assert abs(treino[COL_ALVO].mean() - teste[COL_ALVO].mean()) < 1 / 200


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


def test_separar_x_y_tira_id_alvo_e_colunas_pedidas(df):
    X, y = separar_x_y(df)
    assert not {COL_ID, COL_ALVO} & set(X.columns) and X.shape[1] == df.shape[1] - 2
    assert y.equals(df[COL_ALVO])
    X2, _ = separar_x_y(df, excluir=["x0"])
    assert "x0" not in X2.columns and X2.shape[1] == X.shape[1] - 1


def test_cli_grava_treino_e_teste(tmp_path, df, monkeypatch):
    df.to_csv(tmp_path / "dados.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["divisao.py", "--entrada", str(tmp_path / "dados.csv"), "--saida-dir",
                                      str(tmp_path / "saida")])
    import divisao
    divisao.main()
    assert len(pd.read_csv(tmp_path / "saida" / "treino.csv")) == 800
    assert len(pd.read_csv(tmp_path / "saida" / "teste.csv")) == 200
