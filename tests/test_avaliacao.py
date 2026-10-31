import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import avaliacao  # noqa: E402
from avaliacao import METRICAS_TESTE, bootstrap, escolher_vencedor, modos_disponiveis  # noqa: E402
from dados_genericos import tabela_generica  # noqa: E402
from divisao import COL_ID, dividir, separar_x_y  # noqa: E402
from validacao import _metricas, comparar_modelos, gravar  # noqa: E402

MODOS = ("nenhum", "smote")


def mini_catalogo():
    return {
        "RL": (LogisticRegression(max_iter=500), {"modelo__C": [0.1, 1.0]}),
        "DT": (DecisionTreeClassifier(random_state=0), {"modelo__max_depth": [3, None]}),
    }


@pytest.fixture(scope="module")
def resultados(tmp_path_factory):
    """Pasta com o que `validacao.py` grava (dois modos, dois modelos) e o CSV do teste, como numa execução real."""
    pasta = tmp_path_factory.mktemp("resultados")
    treino, teste = dividir(tabela_generica(n=500, prevalencia=0.20, seed=21))
    teste.to_csv(pasta / "teste.csv", index=False)
    X, y = separar_x_y(treino)
    for modo in MODOS:
        tabela, oof, finais = comparar_modelos(X, y, treino[COL_ID], desbalanceamento=modo, mini_catalogo=mini_catalogo(),
                                               n_dobras=3, n_jobs=1, verbose=False, devolver_modelos=True)
        gravar(tabela, oof, finais, pasta, "enxuta", modo)
    return pasta


@pytest.fixture(scope="module")
def avaliado(resultados):
    return avaliacao.avaliar(resultados, resultados / "teste.csv", n_bootstrap=200, seed=1)


def tabela_validacao(f1, auroc, modelos=("RL", "DT")):
    return pd.DataFrame({"f1": f1, "auroc": auroc}, index=pd.Index(modelos, name="modelo"))


def test_vencedor_e_escolhido_antes_de_o_teste_ser_lido(resultados, monkeypatch):
    ordem = []
    escolher, carregar = avaliacao.escolher_vencedor, avaliacao.carregar
    monkeypatch.setattr(avaliacao, "escolher_vencedor", lambda c: (ordem.append("escolher"), escolher(c))[1])
    monkeypatch.setattr(avaliacao, "carregar", lambda caminho: (ordem.append("carregar"), carregar(caminho))[1])
    avaliacao.avaliar(resultados, resultados / "teste.csv", n_bootstrap=10)
    assert ordem == ["escolher", "carregar"]


def test_escolher_vencedor_pelo_f1_e_desempate_pela_auroc():
    assert escolher_vencedor({"nenhum": tabela_validacao([0.7, 0.8], [0.9, 0.8])}) == ("DT", "nenhum")
    empate = {"nenhum": tabela_validacao([0.8, 0.6], [0.85, 0.9]), "smote": tabela_validacao([0.6, 0.8], [0.9, 0.95])}
    assert escolher_vencedor(empate) == ("DT", "smote")  # F1 empatado entre modos: vence a maior AUROC
    total = {"smote": tabela_validacao([0.8], [0.9], ["RL"]), "nenhum": tabela_validacao([0.8], [0.9], ["RL"])}
    assert escolher_vencedor(total) == ("RL", "nenhum")  # empate total: ordem de DESBALANCEAMENTO
    with pytest.raises(ValueError, match="validacao"):
        escolher_vencedor({})


def test_tabela_do_teste_tem_todos_os_modelos_e_as_metricas_conferem(resultados, avaliado):
    tabela = avaliado["tabela_teste"]
    assert len(tabela) == len(MODOS) * 2 and tabela["vencedor"].sum() == 1
    assert list(tabela.columns) == ["modo", "modelo", *METRICAS_TESTE, "f1_validacao", "vencedor"]
    assert tabela[list(METRICAS_TESTE)].apply(lambda c: c.between(0, 1).all()).all()
    teste = pd.read_csv(resultados / "teste.csv")
    X, y = separar_x_y(teste)
    for _, linha in tabela.iterrows():  # recalculado do zero a partir do modelo salvo
        modelo = avaliacao.carregar_modelo(resultados, "enxuta", linha["modo"], linha["modelo"])
        esperado = _metricas(y, modelo.predict(X), modelo.predict_proba(X)[:, 1])
        for m in METRICAS_TESTE:
            assert linha[m] == pytest.approx(esperado[m]), (linha["modo"], linha["modelo"], m)


def test_vencedor_da_tabela_e_o_da_validacao(avaliado):
    modelo, modo = avaliado["vencedor"]
    linha = avaliado["tabela_teste"].query("vencedor").iloc[0]
    assert (linha["modelo"], linha["modo"]) == (modelo, modo)
    assert escolher_vencedor(avaliado["comparacoes"]) == (modelo, modo)


def test_bootstrap_contem_a_nota_pontual_e_e_reprodutivel():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40), np.zeros(160)].astype(int)
    proba = np.clip(0.3 * y + rng.uniform(0, 0.7, len(y)), 0, 1)
    previsao = (proba > 0.5).astype(int)
    pontual = _metricas(y, previsao, proba)
    a = bootstrap(y, previsao, proba, n=300, seed=5)
    assert a == bootstrap(y, previsao, proba, n=300, seed=5) and a != bootstrap(y, previsao, proba, n=300, seed=6)
    for m, (inf, sup) in a.items():
        assert inf <= pontual[m] <= sup and 0 <= inf <= sup <= 1, m


def test_bootstrap_descarta_reamostragens_com_uma_classe_so():
    y, proba = np.array([1, 0, 0, 0, 0, 0]), np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.2])
    ic = bootstrap(y, (proba > 0.5).astype(int), proba, n=50, seed=0)  # sem o descarte, a AUROC levantaria erro
    assert ic["auroc"] == (1.0, 1.0)


def test_bootstrap_com_uma_classe_so_no_teste_levanta_erro():
    y = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="com e sem sepse"):
        bootstrap(y, y, np.linspace(0, 1, 50), n=10)


def test_gravar_cria_figuras_tabela_e_resumo(resultados, avaliado):
    resumo = avaliacao.gravar(avaliado, resultados, "enxuta")
    for nome in ([f"roc_oof_enxuta_{m}.png" for m in MODOS] + [f"roc_teste_enxuta_{m}.png" for m in MODOS]
                 + ["confusao_vencedor_enxuta.png", "avaliacao_teste_enxuta.csv", "vencedor_enxuta.json"]):
        assert (resultados / nome).stat().st_size > 0, nome
    lido = json.loads((resultados / "vencedor_enxuta.json").read_text(encoding="utf-8"))
    assert lido == resumo and (lido["modelo"], lido["modo"]) == avaliado["vencedor"]
    assert set(lido["teste"]) == set(METRICAS_TESTE) == set(lido["teste_ic95"])
    assert sum(lido["confusao"].values()) == len(avaliado["y_teste"])


def test_sem_resultados_da_validacao_levanta_erro(tmp_path, resultados):
    assert modos_disponiveis(tmp_path, "enxuta") == []
    assert modos_disponiveis(resultados, "completa") == []
    assert modos_disponiveis(resultados, "enxuta") == list(MODOS)
    with pytest.raises(FileNotFoundError, match="validacao"):
        avaliacao.avaliar(tmp_path, resultados / "teste.csv")


def test_cli(resultados, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["avaliacao.py", "--resultados-dir", str(resultados), "--teste",
                                      str(resultados / "teste.csv"), "--n-bootstrap", "20"])
    avaliacao.main()
    saida = capsys.readouterr().out
    assert "vencedor" in saida and "intervalo de 95%" in saida


def test_comparacao_sem_modelos_salvos_levanta_erro(tmp_path):
    (tmp_path / "comparacao_enxuta_nenhum.csv").write_text("modelo,f1,auroc\nRL,0.5,0.5\n")
    with pytest.raises(FileNotFoundError, match="desbalanceamento nenhum"):
        modos_disponiveis(tmp_path, "enxuta")
