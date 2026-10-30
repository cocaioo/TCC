import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validacao  # noqa: E402
from gerador_seed import gerar_seed  # noqa: E402
from modelos import construir_pipeline  # noqa: E402
from preprocessamento import COL_ID, construir_preprocessador, dividir, separar_x_y  # noqa: E402
from validacao import (  # noqa: E402
    METRICAS, _busca, _escolher_melhor, _metricas, _preparar, avaliar_modelo, comparar_modelos, gravar,
)

# Categoria rara que cai só na dobra de validação vira "zeros" no one-hot (por desenho); o aviso inunda a saída dos testes.
pytestmark = pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")


def mini_catalogo():
    return {
        "RL": (LogisticRegression(max_iter=500), {"modelo__C": [0.1, 1.0]}),
        "DT": (DecisionTreeClassifier(random_state=0), {"modelo__max_depth": [3, None]}),
    }


@pytest.fixture(scope="module")
def treino():
    df, _ = dividir(gerar_seed(n=1000, prevalencia=0.20, seed=11))
    X, y = separar_x_y(df)
    X_s, _, y_s, _, id_s, _ = train_test_split(X, y, df[COL_ID], train_size=300, stratify=y, random_state=0)
    return X_s, y_s, id_s


def rodar(X, y, ids=None, **kwargs):
    return comparar_modelos(X, y, ids, mini_catalogo=mini_catalogo(), n_dobras=3, n_jobs=1, verbose=False, **kwargs)


def test_tabela_e_oof_tem_o_formato_certo(treino):
    X, y, ids = treino
    tabela, oof = rodar(X, y, ids)
    assert list(tabela.index) == ["RL", "DT"]
    for coluna in ("f1", "auroc", "precisao", "recall", "acuracia", "f1_busca_simples"):
        assert tabela[coluna].between(0, 1).all(), coluna
    assert set(METRICAS) <= set(tabela.columns) | {"f1_busca_simples"}
    assert len(oof) == len(y) and not oof[["RL", "DT"]].isna().any().any()
    assert oof[["RL", "DT"]].apply(lambda c: c.between(0, 1).all()).all()
    assert (oof[COL_ID].to_numpy() == ids.to_numpy()).all() and (oof["sepse_tardia"].to_numpy() == y.to_numpy()).all()


def test_oof_nao_vaza_com_alvo_aleatorio(treino):
    """Com o alvo embaralhado não há nada a aprender: uma árvore sem limite de profundidade decora o treino, então
    qualquer vazamento da dobra de validação para o ajuste faria a AUROC fora da dobra subir muito acima de 0,5."""
    X, y, ids = treino
    y_ruido = np.random.RandomState(3).permutation(y.to_numpy())
    y_ruido = type(y)(y_ruido, index=y.index)
    cat = {"DT": (DecisionTreeClassifier(random_state=0), {"modelo__max_depth": [None]})}
    tabela, oof = comparar_modelos(X, y_ruido, ids, mini_catalogo=cat, n_dobras=3, n_jobs=1, verbose=False)
    assert roc_auc_score(oof["sepse_tardia"], oof["DT"]) < 0.65
    assert tabela.loc["DT", "auroc"] < 0.65


def test_mesma_semente_mesmo_resultado(treino):
    X, y, ids = treino
    t1, o1 = rodar(X, y, ids, seed=5)
    t2, o2 = rodar(X, y, ids, seed=5)
    assert np.allclose(o1[["RL", "DT"]], o2[["RL", "DT"]])
    assert np.allclose(t1[["f1", "auroc"]], t2[["f1", "auroc"]])


def test_resultado_nao_depende_do_indice_das_linhas(treino):
    """A probabilidade fora da dobra tem que ficar na linha do paciente certo, mesmo com o índice embaralhado."""
    X, y, ids = treino
    ordem = np.random.RandomState(0).permutation(len(X))
    X_b, y_b, ids_b = X.iloc[ordem], y.iloc[ordem], ids.iloc[ordem]  # mantém o índice original, agora fora de ordem
    _, o_ordenado = rodar(X_b.reset_index(drop=True), y_b.reset_index(drop=True), ids_b.reset_index(drop=True), seed=1)
    _, o_embaralhado = rodar(X_b, y_b, ids_b, seed=1)
    assert np.allclose(o_ordenado[["RL", "DT"]], o_embaralhado[["RL", "DT"]])
    assert (o_embaralhado[COL_ID].to_numpy() == ids_b.to_numpy()).all()


def test_hiperparametro_invalido_para_a_execucao(treino):
    X, y, ids = treino
    cat = {"RL": (LogisticRegression(), {"modelo__nao_existe": [1, 2]})}
    with pytest.raises(ValueError):  # error_score="raise": nada de NaN calado
        comparar_modelos(X, y, ids, mini_catalogo=cat, n_dobras=3, n_jobs=1, verbose=False)


def test_entradas_invalidas(treino):
    X, y, ids = treino
    with pytest.raises(ValueError, match="desconhecidos"):
        rodar(X, y, ids, modelos=["XYZ"])
    with pytest.raises(ValueError, match="linhas"):
        rodar(X.iloc[:10], y, ids)


# --- Os números reportados: conferidos à mão, e não só "entre 0 e 1" ------------------------------------


def test_metricas_conferidas_a_mao():
    y = [1, 1, 1, 0, 0, 0, 0]
    previsao = [1, 0, 0, 1, 0, 0, 0]  # VP = 1, FN = 2, FP = 1, VN = 3
    proba = [0.9, 0.4, 0.3, 0.6, 0.2, 0.1, 0.05]
    m = _metricas(y, previsao, proba)
    assert m["precisao"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(0.4)
    assert m["especificidade"] == pytest.approx(3 / 4)  # VN / (VN + FP)
    assert m["acuracia"] == pytest.approx(4 / 7)
    assert m["auroc"] == pytest.approx(10 / 12)  # 10 dos 12 pares (positivo, negativo) em ordem certa


def test_escolher_melhor_desempata_pela_auroc():
    assert _escolher_melhor({"mean_test_f1": [0.5, 0.7, 0.7], "mean_test_auroc": [0.9, 0.6, 0.8]}) == 2
    assert _escolher_melhor({"mean_test_f1": [0.0, 0.0, 0.0], "mean_test_auroc": [0.5, 0.7, 0.6]}) == 1  # tudo 0
    assert _escolher_melhor({"mean_test_f1": [0.3, 0.3], "mean_test_auroc": [0.7, 0.7]}) == 0  # empate total
    assert _escolher_melhor({"mean_test_f1": [0.2, 0.9], "mean_test_auroc": [0.99, 0.5]}) == 1  # o F1 manda


def test_oof_e_informativo_com_o_alvo_real(treino):
    """Com sinal de verdade, a probabilidade fora da dobra tem que separar as classes. Pega probabilidades trocadas de
    paciente, a coluna da classe errada (AUROC invertida) e o `oof` desalinhado."""
    X, y, ids = treino
    _, oof = rodar(X, y, ids, modelos=["RL"])
    assert roc_auc_score(oof["sepse_tardia"], oof["RL"]) > 0.8


def test_linha_da_tabela_bate_com_as_dobras_e_a_busca(treino):
    X, y, ids = treino
    grade = {"modelo__C": [0.001, 0.01, 0.1, 1.0]}
    cat = {"RL": (LogisticRegression(max_iter=500), grade)}
    tabela, _ = comparar_modelos(X, y, ids, mini_catalogo=cat, n_dobras=3, n_jobs=1, verbose=False)
    linha = tabela.loc["RL"]

    prep = construir_preprocessador(X)
    r = avaliar_modelo(construir_pipeline(prep, LogisticRegression(max_iter=500)), grade, X, y, n_dobras=3)
    por_dobra = r["aninhada"]["por_dobra"]
    assert len(por_dobra) == 3  # número de dobras externas = n_dobras
    assert linha["f1"] == pytest.approx(por_dobra["f1"].mean())
    assert linha["f1_desvio"] == pytest.approx(por_dobra["f1"].std(ddof=1))
    assert linha["otimismo_f1"] == pytest.approx(linha["f1_busca_simples"] - linha["f1"])
    assert linha["melhores_params"] == r["simples"]["params"] and "modelo__C" in linha["melhores_params"]

    # o vencedor da busca simples é de fato o de maior F1 da grade (e não o primeiro, nem o de outra métrica)
    busca = _busca(construir_pipeline(prep, LogisticRegression(max_iter=500)), grade, 3, 42, 1).fit(X, y)
    assert linha["f1_busca_simples"] == pytest.approx(busca.cv_results_["mean_test_f1"].max())


def test_catalogo_real_recebe_os_argumentos_certos(treino, monkeypatch):
    X, y, ids = treino
    chamadas = []

    def espiao(*args, **kwargs):
        chamadas.append((args, kwargs))
        return mini_catalogo()

    monkeypatch.setattr(validacao, "catalogo", espiao)
    comparar_modelos(X, y, ids, grade="enxuta", desbalanceamento="class_weight", modelos=["RL"], seed=7,
                     n_dobras=3, n_jobs=1, verbose=False)
    (args, _), = chamadas
    assert args == ("enxuta", "class_weight", 7)


def test_modelos_repetidos_ou_em_texto_sao_normalizados(treino):
    X, y, ids = treino
    tabela, oof = rodar(X, y, ids, modelos=["RL", "RL"])
    assert list(tabela.index) == ["RL"] and list(oof.columns) == [COL_ID, "sepse_tardia", "RL"]
    assert list(rodar(X, y, ids, modelos="RL")[0].index) == ["RL"]  # "RL" não vira ["R", "L"]


# --- Entradas que não podem passar calado ------------------------------------------------------------------


def test_poucos_positivos_levanta_erro_em_vez_de_devolver_f1_zero(treino):
    X, _, ids = treino
    y_raro = pd.Series(np.zeros(len(X), dtype=int))
    y_raro.iloc[:5] = 1  # 5 positivos < 3 dobras ao quadrado (9)
    with pytest.raises(ValueError, match="pelo menos"):
        rodar(X, y_raro, ids)


def test_rotulos_que_nao_sao_0_e_1_levantam_erro(treino):
    X, y, _ = treino
    with pytest.raises(ValueError, match="0 e 1"):
        _preparar(X, y + 1, 3)  # rótulos {1, 2} tratariam a classe NEGATIVA como positiva
    with pytest.raises(ValueError, match="0 e 1"):
        _preparar(X, y.map({0: "nao", 1: "sim"}), 3)


def test_preparar_aceita_y_em_array_e_exige_dataframe(treino):
    X, y, _ = treino
    X2, y2 = _preparar(X, y.to_numpy(), 3)
    assert isinstance(y2, pd.Series) and (y2.to_numpy() == y.to_numpy()).all() and list(X2.index) == list(range(len(X)))
    with pytest.raises(TypeError, match="DataFrame"):
        _preparar(X.to_numpy(), y, 3)


# --- Linha de comando ------------------------------------------------------------------------------------------


def test_cli_grava_os_dois_arquivos(tmp_path, monkeypatch):
    gerar_seed(n=300, prevalencia=0.20, seed=21).to_csv(tmp_path / "treino.csv", index=False)
    saida = tmp_path / "resultados"
    monkeypatch.setattr(sys, "argv", ["validacao.py", "--treino", str(tmp_path / "treino.csv"), "--saida-dir",
                                      str(saida), "--modelos", "NB", "--n-jobs", "1"])
    validacao.main()
    comparacao = pd.read_csv(saida / "comparacao_enxuta_class_weight.csv", index_col="modelo")
    oof = pd.read_csv(saida / "oof_enxuta_class_weight.csv")
    assert list(comparacao.index) == ["NB"] and 0 <= comparacao.loc["NB", "f1"] <= 1
    assert len(oof) == 300 and oof["NB"].between(0, 1).all() and oof[COL_ID].is_unique
    assert [p.name for p in (saida / "modelos_enxuta_class_weight").iterdir()] == ["NB.joblib"]


# --- SMOTE ----------------------------------------------------------------------------------------------------


def test_comparar_modelos_com_smote_mantem_o_oof_alinhado(treino, monkeypatch):
    X, y, ids = treino
    montados = []
    original = validacao.construir_pipeline

    def espiao(prep, estimador, desbalanceamento="nenhum", seed=42):
        pipe = original(prep, estimador, desbalanceamento, seed)
        montados.append(pipe)
        return pipe

    monkeypatch.setattr(validacao, "construir_pipeline", espiao)
    tabela, oof = rodar(X, y, ids, desbalanceamento="smote", seed=4)
    assert all("smote" in pipe.named_steps and pipe["smote"].random_state == 4 for pipe in montados)
    assert len(montados) == 2 and not oof[["RL", "DT"]].isna().any().any()
    assert (oof[COL_ID].to_numpy() == ids.to_numpy()).all()
    assert roc_auc_score(oof["sepse_tardia"], oof["RL"]) > 0.8  # continua informativo e alinhado


# --- Modelos finais (passo 3) -------------------------------------------------------------------------------------


def test_modelos_finais_sao_os_vencedores_reajustados_e_sobrevivem_ao_disco(treino, tmp_path):
    X, y, ids = treino
    tabela, oof, finais = rodar(X, y, ids, devolver_modelos=True)
    assert set(finais) == {"RL", "DT"}
    for nome, pipe in finais.items():  # é o vencedor da busca simples, em todos os hiperparâmetros da grade
        melhores = tabela.loc[nome, "melhores_params"]
        assert melhores and {k: pipe.get_params()[k] for k in melhores} == melhores
    gravar(tabela, oof, finais, tmp_path, "enxuta", "nenhum")
    for nome, pipe in finais.items():
        recarregado = joblib.load(tmp_path / "modelos_enxuta_nenhum" / f"{nome}.joblib")
        assert np.allclose(recarregado.predict_proba(X), pipe.predict_proba(X))
    assert (tmp_path / "comparacao_enxuta_nenhum.csv").exists() and (tmp_path / "oof_enxuta_nenhum.csv").exists()


def test_gravar_apaga_modelos_de_rodada_anterior(treino, tmp_path):
    X, y, ids = treino
    tabela, oof, finais = rodar(X, y, ids, modelos=["RL"], devolver_modelos=True)
    pasta = tmp_path / "modelos_enxuta_nenhum"
    pasta.mkdir()
    (pasta / "DT.joblib").write_text("modelo velho")
    gravar(tabela, oof, finais, tmp_path, "enxuta", "nenhum")
    assert sorted(p.name for p in pasta.iterdir()) == ["RL.joblib"]


def test_especificidade_entra_na_tabela(treino):
    X, y, ids = treino
    tabela, _ = rodar(X, y, ids)
    assert tabela["especificidade"].between(0, 1).all()
