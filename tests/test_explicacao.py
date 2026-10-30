import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import explicacao  # noqa: E402
from explicacao import explicar, importancia_global, prever, separar_pipeline  # noqa: E402
from gerador_seed import gerar_seed  # noqa: E402
from modelos import catalogo, construir_pipeline  # noqa: E402
from preprocessamento import COL_ID, construir_preprocessador, dividir, separar_x_y  # noqa: E402
from validacao import _metricas  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")


@pytest.fixture(scope="module")
def pasta(tmp_path_factory):
    """O que `avaliacao.py` deixa pronto: o vencedor salvo (uma árvore, modo smote), seu resumo e os CSVs."""
    pasta = tmp_path_factory.mktemp("resultados")
    treino, teste = dividir(gerar_seed(n=400, prevalencia=0.20, seed=31))
    treino.to_csv(pasta / "treino.csv", index=False)
    teste.iloc[:30].to_csv(pasta / "teste.csv", index=False)  # poucos pacientes: o SHAP por permutação é lento
    X, y = separar_x_y(treino)
    estimador = catalogo(desbalanceamento="smote")["DT"][0].set_params(max_depth=4)
    pipe = construir_pipeline(construir_preprocessador(X), estimador, "smote", seed=1).fit(X, y)
    (pasta / "modelos_enxuta_smote").mkdir()
    joblib.dump(pipe, pasta / "modelos_enxuta_smote" / "DT.joblib")
    X_teste, y_teste = separar_x_y(teste.iloc[:30])
    metricas = _metricas(y_teste, pipe.predict(X_teste), pipe.predict_proba(X_teste)[:, 1])  # o que avaliacao.py gravaria
    resumo = {"modelo": "DT", "modo": "smote", "grade": "enxuta", "limiar": 0.5, "validacao_aninhada": metricas,
              "teste": metricas, "teste_ic95": {m: [0.4, 0.6] for m in metricas},
              "confusao": {"vn": 1, "fp": 1, "fn": 1, "vp": 1}}
    (pasta / "vencedor_enxuta.json").write_text(json.dumps(resumo), encoding="utf-8")
    return pasta


@pytest.fixture(scope="module")
def executado(pasta):
    return explicacao.executar(pasta, pasta / "treino.csv", pasta / "teste.csv", seed=0)


def test_separar_pipeline_deixa_o_smote_de_fora(pasta):
    pipe = joblib.load(pasta / "modelos_enxuta_smote" / "DT.joblib")
    prep, modelo = separar_pipeline(pipe)
    assert [nome for nome, _ in prep.steps] == ["estrutural", "log1p", "colunas", "float"]
    assert modelo is pipe["modelo"]
    X, _ = separar_x_y(pd.read_csv(pasta / "teste.csv"))
    assert len(prep.transform(X)) == len(X)  # nenhuma linha sintética na transformação


def test_shap_soma_exatamente_a_probabilidade_prevista(pasta, executado):
    pipe = joblib.load(pasta / "modelos_enxuta_smote" / "DT.joblib")
    X, _ = separar_x_y(pd.read_csv(pasta / "teste.csv"))
    e = executado["explicacao"]
    reconstruida = e.base_values + e.values.sum(axis=1)
    assert np.allclose(reconstruida, pipe.predict_proba(X)[:, 1], atol=1e-8)


def test_shap_nao_depende_do_valor_de_variavel_que_o_modelo_ignora(pasta):
    """Controle: uma árvore rasa usa poucas variáveis; as que ela não usa têm SHAP zero em todos os pacientes."""
    pipe = joblib.load(pasta / "modelos_enxuta_smote" / "DT.joblib")
    treino, teste = pd.read_csv(pasta / "treino.csv"), pd.read_csv(pasta / "teste.csv")
    e = explicar(pipe, separar_x_y(treino)[0], separar_x_y(teste)[0], seed=0)
    usadas = set(np.asarray(e.feature_names)[pipe["modelo"].feature_importances_ > 0])
    for nome, coluna in zip(e.feature_names, e.values.T):
        if nome not in usadas:
            assert np.allclose(coluna, 0), nome
    assert usadas and any(not np.allclose(c, 0) for n, c in zip(e.feature_names, e.values.T) if n in usadas)


def test_importancia_global_tem_uma_linha_por_variavel_e_vem_ordenada(executado):
    e, tabela = executado["explicacao"], executado["importancia"]
    assert len(tabela) == e.values.shape[1] and tabela["variavel"].is_unique
    assert tabela["shap_medio_absoluto"].is_monotonic_decreasing
    assert tabela.equals(importancia_global(e))


def test_figuras_e_importancia_gravadas(pasta, executado):
    for nome in ("importancia_shap_enxuta.csv", "shap_resumo_enxuta.png", "shap_caso_enxuta.png"):
        assert (pasta / nome).stat().st_size > 0, nome


def test_modelo_final_salvo_e_igual_ao_vencedor(pasta, executado):
    vencedor = joblib.load(pasta / "modelos_enxuta_smote" / "DT.joblib")
    final = joblib.load(pasta / "modelo_final.joblib")
    X, _ = separar_x_y(pd.read_csv(pasta / "teste.csv"))
    assert np.allclose(final.predict_proba(X), vencedor.predict_proba(X))


def test_json_do_modelo_final(pasta, executado):
    meta = json.loads((pasta / "modelo_final.json").read_text(encoding="utf-8"))
    assert meta == executado["metadados"]
    assert (meta["algoritmo"], meta["desbalanceamento"], meta["limiar"]) == ("DT", "smote", 0.5)
    X, _ = separar_x_y(pd.read_csv(pasta / "treino.csv"))
    assert meta["colunas_entrada"] == list(X.columns) and COL_ID not in meta["colunas_entrada"]
    assert meta["hiperparametros"]["max_depth"] == 4 and set(meta["hiperparametros"]) == {"max_depth",
                                                                                          "min_samples_leaf"}
    for chave in ("metricas_validacao_aninhada", "metricas_teste", "metricas_teste_ic95", "data", "versoes"):
        assert meta[chave], chave


def test_prever_devolve_probabilidades_por_paciente(pasta, executado):
    teste = pd.read_csv(pasta / "teste.csv")
    p = prever(teste.drop(columns=["sepse_tardia", "dias_ate_dx_sepse"]), pasta)
    vencedor = joblib.load(pasta / "modelos_enxuta_smote" / "DT.joblib")
    assert list(p.index) == list(teste[COL_ID]) and p.name == "probabilidade_sepse"
    assert np.allclose(p.to_numpy(), vencedor.predict_proba(separar_x_y(teste)[0])[:, 1])


def test_prever_recusa_coluna_faltando_e_avisa_coluna_a_mais(pasta, executado):
    teste = pd.read_csv(pasta / "teste.csv")
    with pytest.raises(ValueError, match="pcr_pre"):
        prever(teste.drop(columns="pcr_pre"), pasta)
    with pytest.warns(UserWarning, match="ignoradas"):
        p = prever(teste.assign(coluna_nova=1), pasta)  # sepse_tardia e coluna_nova sobram
    assert len(p) == len(teste)


def test_sem_avaliacao_levanta_erro(tmp_path, pasta):
    with pytest.raises(FileNotFoundError, match="avaliacao"):
        explicacao.executar(tmp_path, pasta / "treino.csv", pasta / "teste.csv")


def test_escolher_caso_pega_o_alerta_de_probabilidade_mediana():
    assert explicacao.escolher_caso([0.1, 0.9, 0.6, 0.99, 0.3]) == 1  # alertas 0,6 / 0,9 / 0,99: a mediana é 0,9
    assert explicacao.escolher_caso([0.1, 0.7, 0.6]) == 2  # número par de alertas: o menor dos dois do meio
    assert explicacao.escolher_caso([0.1, 0.4, 0.2]) == 1  # nenhum alerta: o de maior probabilidade


def test_modelo_salvo_diferente_do_avaliado_levanta_erro(pasta, tmp_path):
    """Se validacao.py rodou de novo e avaliacao.py não, o modelo salvo não é o que foi medido no teste."""
    for nome in ("treino.csv", "teste.csv", "vencedor_enxuta.json"):
        (tmp_path / nome).write_bytes((pasta / nome).read_bytes())
    (tmp_path / "modelos_enxuta_smote").mkdir()
    pipe = joblib.load(pasta / "modelos_enxuta_smote" / "DT.joblib")
    X, y = separar_x_y(pd.read_csv(pasta / "treino.csv"))
    outro = pipe.set_params(modelo__max_depth=1).fit(X, y)  # "nova rodada": outro modelo no mesmo arquivo
    joblib.dump(outro, tmp_path / "modelos_enxuta_smote" / "DT.joblib")
    with pytest.raises(ValueError, match="não é o avaliado"):
        explicacao.executar(tmp_path, tmp_path / "treino.csv", tmp_path / "teste.csv")
