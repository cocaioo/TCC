import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gerador_seed import gerar_seed  # noqa: E402
from preprocessamento import (  # noqa: E402
    COL_ALVO, COL_EXCLUIDA, COL_ID, SUFIXOS_EXAME, aplicar_log1p, auditar, carregar,
    construir_preprocessador, dividir, separar_x_y, tratar_ausentes_estruturais,
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


# --- Passo 3: pré-processador ----------------------------------------------------------


@pytest.fixture(scope="module")
def X_treino_teste(df):
    treino, teste = dividir(df, seed=42)
    return separar_x_y(treino)[0], separar_x_y(teste)[0]


def test_matriz_sem_nan_com_nomes_unicos_e_indice_preservado(X_treino_teste):
    X_tr, X_te = X_treino_teste
    prep = construir_preprocessador(X_tr)
    A, B = prep.fit_transform(X_tr), prep.transform(X_te)
    assert not A.isna().any().any() and not B.isna().any().any()
    assert A.columns.is_unique and list(A.columns) == list(B.columns)
    assert A.index.equals(X_tr.index) and B.index.equals(X_te.index)  # alinhado com y
    assert (A.dtypes == float).all()


def test_colunas_proibidas_nao_chegam_ao_modelo(df):
    treino, _ = dividir(df, seed=42)
    X, _ = separar_x_y(treino)  # separar_x_y tira id, alvo e dias_ate_dx_sepse
    saida = construir_preprocessador(X).fit_transform(X)
    for proibida in (COL_ID, COL_ALVO, COL_EXCLUIDA):
        assert not any(c.startswith(proibida) for c in saida.columns)


def test_valores_aprendidos_vem_so_do_treino(X_treino_teste):
    X_tr, X_te = X_treino_teste
    prep = construir_preprocessador(X_tr).fit(X_tr)
    imputador = prep.named_steps["colunas"].named_transformers_["num"].named_steps["imputar"]
    medianas = pd.Series(imputador.statistics_, index=imputador.feature_names_in_)
    esperado = aplicar_log1p(X_tr)[medianas.index].median()
    pd.testing.assert_series_equal(medianas, esperado, check_names=False)

    # transformar (mesmo um teste extremo) não altera o que foi aprendido
    antes = prep.transform(X_tr)
    extremo = X_te.copy()
    extremo[["pcr_pre", "pct_pre"]] = 1e6
    prep.transform(extremo)
    pd.testing.assert_frame_equal(antes, prep.transform(X_tr))


def test_hemocultura_ausente_estrutural_vira_categoria():
    X = pd.DataFrame({
        "hemocultura_resultado": ["negativa", None, "positiva", "positiva"],
        "hemocultura_agente": [None, None, None, "scn"],
        "perfil_sensibilidade": [None, None, None, "sensivel"],
    })
    t = tratar_ausentes_estruturais(X)
    assert t["hemocultura_resultado"].tolist() == ["negativa", "nao_realizada", "positiva", "positiva"]
    assert t["hemocultura_agente"].tolist() == ["nao_se_aplica", "nao_realizada", "nao_informado", "scn"]
    assert t["perfil_sensibilidade"].tolist() == ["nao_se_aplica", "nao_realizada", "nao_informado", "sensivel"]
    assert X["hemocultura_agente"].isna().sum() == 3  # a entrada não é alterada


def test_indicador_de_ausencia_so_nos_exames_acima_do_limiar(X_treino_teste):
    X_tr, _ = X_treino_teste
    exames = [c for c in X_tr.columns if c.endswith(SUFIXOS_EXAME)]
    taxa = X_tr[exames].isna().mean()

    saida = construir_preprocessador(X_tr, limiar_indicador=0.20).fit_transform(X_tr)
    indicadores = {c.removeprefix("missingindicator_") for c in saida.columns if c.startswith("missingindicator_")}
    assert indicadores == set(taxa[taxa >= 0.20].index)
    assert "pct_pre" in indicadores and "plaquetas_pre" not in indicadores

    sem = construir_preprocessador(X_tr, limiar_indicador=1.0).fit_transform(X_tr)
    assert not any(c.startswith("missingindicator_") for c in sem.columns)
    todos = construir_preprocessador(X_tr, limiar_indicador=0.0).fit_transform(X_tr)
    assert sum(c.startswith("missingindicator_") for c in todos.columns) == len(exames)


def test_padroniza_numericas_e_preserva_binarias(X_treino_teste):
    X_tr, _ = X_treino_teste
    saida = construir_preprocessador(X_tr).fit_transform(X_tr)
    for coluna in ("idade_gestacional_sem", "peso_nascer_g", "fc", "pcr_pre", "pct_pos24h"):
        assert saida[coluna].mean() == pytest.approx(0, abs=1e-9)
        assert saida[coluna].std(ddof=0) == pytest.approx(1, abs=1e-9)
    for coluna in ("cvc", "vm_invasiva", "ecn"):
        np.testing.assert_array_equal(saida[coluna], X_tr[coluna].astype(float))


def test_log1p_reduz_assimetria_e_rejeita_valor_invalido(X_treino_teste):
    X_tr, _ = X_treino_teste
    t = aplicar_log1p(X_tr)
    for coluna in ("pcr_pre", "pct_pre", "lactato_pre"):
        assert t[coluna].skew() < X_tr[coluna].skew()
    pd.testing.assert_series_equal(t["glicemia_pre"], X_tr["glicemia_pre"])  # fora da lista: intacto

    invalido = X_tr.copy()
    invalido.iloc[0, invalido.columns.get_loc("pcr_pre")] = -999
    with pytest.raises(ValueError, match="log1p"):
        aplicar_log1p(invalido)


def test_categoria_nova_no_teste_vira_zeros(X_treino_teste):
    X_tr, X_te = X_treino_teste
    prep = construir_preprocessador(X_tr).fit(X_tr)
    novo = X_te.copy()
    novo["raca_cor_rn"] = "nova"
    saida = prep.transform(novo)
    assert (saida.filter(like="raca_cor_rn_") == 0).all().all()


def test_ausentes_fora_dos_exames_tambem_sao_tratados(X_treino_teste):
    X_tr, _ = X_treino_teste
    X = X_tr.copy()
    X.loc[X.index[:20], "idade_materna"] = np.nan
    X.loc[X.index[:20], "raca_cor_mae"] = np.nan
    saida = construir_preprocessador(X).fit_transform(X)
    assert not saida.isna().any().any()
    assert "raca_cor_mae_ausente" in saida.columns


def test_preprocessador_nasce_sem_ajuste_e_e_reajustavel(X_treino_teste):
    X_tr, _ = X_treino_teste
    prep = construir_preprocessador(X_tr)
    with pytest.raises(NotFittedError):
        prep.transform(X_tr)
    pd.testing.assert_frame_equal(prep.fit_transform(X_tr), clone(prep).fit_transform(X_tr))
