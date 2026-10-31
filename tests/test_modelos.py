import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline as PipelineSk
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dados_genericos import tabela_generica  # noqa: E402
from modelos import (  # noqa: E402
    DESBALANCEAMENTO, GRADES, MODELOS, SEM_PESOS, SMOTE_VIZINHOS, GradientBoostingBalanceado, XGBClassifierBalanceado, catalogo, construir_pipeline,
    n_combinacoes,
)
from divisao import COL_ALVO, dividir, separar_x_y  # noqa: E402

PREVALENCIA = 0.20


@pytest.fixture(scope="module")
def dados():
    treino, _ = dividir(tabela_generica(n=1000, prevalencia=PREVALENCIA, seed=11))
    X, y = separar_x_y(treino)
    return X, y


@pytest.fixture(scope="module")
def amostra(dados):
    """Cerca de 300 linhas do treino (estratificadas), para os ajustes dos testes ficarem rápidos."""
    X, y = dados
    X_s, _, y_s, _ = train_test_split(X, y, train_size=300, stratify=y, random_state=0)
    return X_s, y_s


@pytest.fixture(scope="module")
def prep():
    """Um pré-processamento qualquer (padronização), só para ocupar o espaço do pipeline nos testes."""
    return PipelineSk([("escalar", StandardScaler())]).set_output(transform="pandas")


def test_catalogo_tem_os_8_modelos_nas_duas_grades():
    for grade in GRADES:
        cat = catalogo(grade)
        assert tuple(cat) == MODELOS and len(cat) == 8
        for estimador, parametros in cat.values():
            assert hasattr(estimador, "fit") and parametros
            assert all(nome.startswith("modelo__") for nome in parametros)


def test_numero_de_combinacoes_das_grades():
    total = {g: sum(n_combinacoes(gr) for _, gr in catalogo(g).values()) for g in GRADES}
    assert total == {"enxuta": 59, "completa": 325}


def test_todo_hiperparametro_de_toda_grade_existe_no_pipeline(prep):
    for grade in GRADES:
        for nome, (estimador, parametros) in catalogo(grade).items():
            combinacoes = list(ParameterGrid(parametros))
            for combinacao in (combinacoes[0], combinacoes[-1]):
                construir_pipeline(prep, estimador).set_params(**combinacao)  # erro de digitação levanta ValueError


def test_cada_pipeline_ajusta_e_devolve_probabilidades(prep, amostra):
    X, y = amostra
    for nome, (estimador, _) in catalogo().items():
        p = construir_pipeline(prep, estimador).fit(X, y).predict_proba(X)
        assert p.shape == (len(X), 2), nome
        assert not np.isnan(p).any(), nome
        assert np.allclose(p.sum(axis=1), 1.0), nome


def test_clone_dos_pipelines_funciona(prep):
    for modo in DESBALANCEAMENTO:
        for nome, (estimador, _) in catalogo(desbalanceamento=modo).items():
            pipe = construir_pipeline(prep, estimador, modo)
            copia = clone(pipe)
            assert type(copia["modelo"]) is type(pipe["modelo"]), nome
            assert copia["modelo"].get_params() == pipe["modelo"].get_params(), nome


def test_class_weight_nos_modelos_que_aceitam():
    cat = catalogo(desbalanceamento="class_weight")
    base = catalogo(desbalanceamento="nenhum")
    for nome in ("RL", "DT", "RF"):
        assert cat[nome][0].get_params()["class_weight"] == "balanced"
    assert isinstance(cat["XGBoost"][0], XGBClassifierBalanceado)
    assert isinstance(cat["GB"][0], GradientBoostingBalanceado)
    for nome in SEM_PESOS:  # sem suporte no scikit-learn: ficam como no modo "nenhum"
        assert cat[nome][0].get_params() == base[nome][0].get_params()


def test_modo_nenhum_nao_pesa_nada():
    cat = catalogo(desbalanceamento="nenhum")
    for nome in ("RL", "DT", "RF"):
        assert cat[nome][0].get_params()["class_weight"] is None
    assert type(cat["XGBoost"][0]) is XGBClassifier
    assert type(cat["GB"][0]) is GradientBoostingClassifier


def test_gradient_boosting_balanceado_aplica_o_peso(prep, amostra):
    X, y = amostra
    Xp = clone(prep).fit_transform(X)  # clone: o `prep` da fixture é compartilhado e precisa continuar sem ajuste
    simples = GradientBoostingClassifier(random_state=0).fit(Xp, y)
    balanceado = GradientBoostingBalanceado(random_state=0).fit(Xp, y)
    p_simples = simples.predict_proba(Xp)[:, 1]
    p_balanceado = balanceado.predict_proba(Xp)[:, 1]
    assert not np.allclose(p_simples, p_balanceado)
    assert p_balanceado.mean() > p_simples.mean()  # a classe rara pesa mais, então as probabilidades sobem


def test_opcoes_invalidas_levantam_erro(prep):
    with pytest.raises(ValueError, match="grade"):
        catalogo(grade="gigante")
    with pytest.raises(ValueError, match="desbalanceamento"):
        catalogo(desbalanceamento="subamostragem")
    with pytest.raises(ValueError, match="desbalanceamento"):
        construir_pipeline(prep, catalogo()["RL"][0], desbalanceamento="subamostragem")


def test_preprocessador_vem_primeiro_e_nao_esta_ajustado(prep, amostra):
    X, y = amostra
    cat = catalogo()
    p1 = construir_pipeline(prep, cat["RL"][0])
    p2 = construir_pipeline(prep, cat["DT"][0])
    assert [nome for nome, _ in p1.steps] == ["escalar", "modelo"]
    p1.fit(X, y)
    check_is_fitted(p1["escalar"])  # controle: o pipeline ajustado aprendeu
    # Ajustar um pipeline não pode contaminar os outros nem o original.
    for objeto in (prep["escalar"], p2["escalar"]):
        with pytest.raises(NotFittedError):
            check_is_fitted(objeto)


def test_espaco_do_preprocessamento_aceita_vazio_ou_um_transformador(amostra):
    X, y = amostra
    estimador = catalogo()["RL"][0]
    vazio = construir_pipeline(None, estimador)
    assert [nome for nome, _ in vazio.steps] == ["modelo"]
    assert np.isfinite(vazio.fit(X, y).predict_proba(X)).all()
    unico = construir_pipeline(StandardScaler(), estimador, "smote")
    assert [nome for nome, _ in unico.steps] == ["preprocessamento", "smote", "modelo"]
    assert np.isfinite(unico.fit(X, y).predict_proba(X)).all()


def test_mesma_semente_mesmas_probabilidades(prep, amostra):
    X, y = amostra
    for nome in MODELOS:
        a = construir_pipeline(prep, catalogo(seed=7)[nome][0]).fit(X, y).predict_proba(X)
        b = construir_pipeline(prep, catalogo(seed=7)[nome][0]).fit(X, y).predict_proba(X)
        assert np.allclose(a, b), nome


def test_alvo_nao_entra_nas_variaveis(dados):
    X, _ = dados
    assert COL_ALVO not in X.columns


def test_valores_reais_das_grades_ajustam_primeira_do_meio_e_ultima(prep, amostra):
    """`set_params` não valida valores: só ajustar pega `n_neighbors=0`, `weights` inexistente, `max_features` inválido."""
    X, y = amostra
    X, y = X.iloc[:150], y.iloc[:150]
    for grade in GRADES:
        for nome, (estimador, parametros) in catalogo(grade, "class_weight").items():
            combinacoes = list(ParameterGrid(parametros))
            for combinacao in (combinacoes[0], combinacoes[len(combinacoes) // 2], combinacoes[-1]):
                pipe = construir_pipeline(prep, estimador).set_params(**combinacao).fit(X, y)
                assert np.isfinite(pipe.predict_proba(X)).all(), (grade, nome, combinacao)


def test_semente_chega_a_todo_modelo_estocastico():
    for semente in (1, 42):
        cat = catalogo(seed=semente)
        for nome in ("RL", "DT", "RF", "GB", "XGBoost", "MLP"):
            assert cat[nome][0].get_params()["random_state"] == semente, nome


def test_parametros_fixos_decididos_no_plano():
    cat = catalogo()
    assert cat["MLP"][0].get_params()["early_stopping"] is True and cat["MLP"][0].get_params()["max_iter"] == 500
    assert cat["RL"][0].get_params()["max_iter"] == 2000
    assert cat["XGBoost"][0].get_params()["n_jobs"] == 1  # o paralelismo fica na busca em grade


def test_gradient_boosting_balanceado_respeita_o_peso_do_usuario(prep, amostra):
    X, y = amostra
    Xp = clone(prep).fit_transform(X)
    w = np.random.RandomState(0).uniform(0.5, 2.0, len(y))
    peso = compute_sample_weight("balanced", y) * w
    esperado = GradientBoostingClassifier(random_state=0).fit(Xp, y, sample_weight=peso)
    obtido = GradientBoostingBalanceado(random_state=0).fit(Xp, y, sample_weight=w)
    assert np.allclose(esperado.predict_proba(Xp), obtido.predict_proba(Xp))


def test_xgboost_balanceado_calcula_o_peso_a_cada_fit(monkeypatch):
    """O peso dos positivos (negativos/positivos) vem do `y` de cada ajuste, e não de um número fixo."""
    capturados = []

    def espiao(self, X, y, sample_weight=None, **kwargs):
        capturados.append(np.asarray(sample_weight))
        return self

    monkeypatch.setattr(XGBClassifier, "fit", espiao)
    X = np.zeros((100, 2))
    for n_positivos in (10, 30):  # duas "dobras" com prevalências diferentes
        y = np.r_[np.ones(n_positivos), np.zeros(100 - n_positivos)].astype(int)
        XGBClassifierBalanceado().fit(X, y)
    (peso_10, peso_30) = capturados
    assert set(peso_10[:10]) == {90 / 10} and set(peso_10[10:]) == {1.0}
    assert set(peso_30[:30]) == {70 / 30} and set(peso_30[30:]) == {1.0}
    XGBClassifierBalanceado().fit(X, y, sample_weight=np.full(100, 2.0))  # o peso do usuário é preservado
    assert set(capturados[-1][:30]) == {2 * 70 / 30}


def test_xgboost_balanceado_exige_positivos():
    with pytest.raises(ValueError, match="positivo"):
        XGBClassifierBalanceado().fit(np.zeros((20, 2)), np.zeros(20, dtype=int))


def test_xgboost_balanceado_sobe_as_probabilidades(prep, amostra):
    X, y = amostra
    Xp = clone(prep).fit_transform(X)
    simples = XGBClassifier(n_estimators=50, random_state=0, n_jobs=1).fit(Xp, y).predict_proba(Xp)[:, 1]
    balanceado = XGBClassifierBalanceado(n_estimators=50, random_state=0, n_jobs=1).fit(Xp, y).predict_proba(Xp)[:, 1]
    assert balanceado.mean() > simples.mean()  # a classe rara pesa mais


def test_grades_devolvidas_nao_compartilham_listas():
    catalogo()["RL"][1]["modelo__C"].append(999)
    assert 999 not in catalogo()["RL"][1]["modelo__C"]


# --- SMOTE ----------------------------------------------------------------------------------------------------


class Espiao(ClassifierMixin, BaseEstimator):
    """Estimador que só registra o que recebe: as classes no `fit` e o número de linhas no `predict_proba`."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.contagem_fit_ = np.bincount(np.asarray(y, dtype=int))
        self.linhas_fit_ = len(X)
        return self

    def predict_proba(self, X):
        self.linhas_predict_ = len(X)
        return np.tile([0.5, 0.5], (len(X), 1))


def test_modo_smote_usa_os_estimadores_sem_pesos():
    smote, nenhum = catalogo(desbalanceamento="smote"), catalogo(desbalanceamento="nenhum")
    for nome in MODELOS:
        assert type(smote[nome][0]) is type(nenhum[nome][0]), nome
        assert smote[nome][0].get_params() == nenhum[nome][0].get_params(), nome


def test_passo_smote_so_existe_no_modo_smote(prep):
    estimador = catalogo()["RL"][0]
    nomes = {modo: [n for n, _ in construir_pipeline(prep, estimador, modo).steps] for modo in DESBALANCEAMENTO}
    assert nomes["smote"] == ["escalar", "smote", "modelo"]
    assert "smote" not in nomes["nenhum"] and "smote" not in nomes["class_weight"]
    smote = construir_pipeline(prep, estimador, "smote", seed=9)["smote"]
    assert isinstance(smote, SMOTE) and smote.k_neighbors == SMOTE_VIZINHOS and smote.random_state == 9


def test_os_8_pipelines_ajustam_no_modo_smote(prep, amostra):
    X, y = amostra
    for nome, (estimador, _) in catalogo(desbalanceamento="smote").items():
        p = construir_pipeline(prep, estimador, "smote").fit(X, y).predict_proba(X)
        assert p.shape == (len(X), 2) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0), nome


def test_smote_reamostra_so_no_fit(prep, amostra):
    X, y = amostra
    pipe = construir_pipeline(prep, Espiao(), "smote").fit(X, y)
    espiao = pipe["modelo"]
    negativos = int((y == 0).sum())
    assert list(espiao.contagem_fit_) == [negativos, negativos]  # classes igualadas no treino
    assert espiao.linhas_fit_ == 2 * negativos > len(X)
    pipe.predict_proba(X.iloc[:37])
    assert espiao.linhas_predict_ == 37  # a previsão recebe só as linhas pedidas, sem exemplos sintéticos


def test_smote_dentro_do_pipeline_nao_vaza_e_o_teste_detectaria_o_vazamento(prep, amostra):
    """Com o alvo embaralhado não há o que aprender. Com o SMOTE dentro do pipeline (só no treino de cada dobra), a
    AUROC fora da dobra de uma árvore que decora fica perto de 0,5. Controle: reamostrar ANTES de dividir põe exemplos
    sintéticos, vizinhos dos positivos de validação, no treino, e a AUROC sobe; isso prova que o teste pegaria o erro."""
    X, y = amostra
    y_ruido = pd.Series(np.random.RandomState(3).permutation(y.to_numpy()), index=y.index)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    arvore = DecisionTreeClassifier(random_state=0)

    pipe = construir_pipeline(prep, arvore, "smote")
    proba = cross_val_predict(pipe, X, y_ruido, cv=cv, method="predict_proba")[:, 1]
    auroc_certo = roc_auc_score(y_ruido, proba)

    Xp = clone(prep).fit_transform(X)
    X_res, y_res = SMOTE(random_state=0).fit_resample(Xp, y_ruido)
    proba_errado = cross_val_predict(arvore, X_res, y_res, cv=cv, method="predict_proba")[:, 1]
    auroc_errado = roc_auc_score(y_res, proba_errado)

    assert auroc_certo < 0.62
    assert auroc_errado > auroc_certo + 0.10


def test_smote_mesma_semente_mesmo_resultado(prep, amostra):
    X, y = amostra
    estimador = catalogo(desbalanceamento="smote")["RL"][0]
    a = construir_pipeline(prep, estimador, "smote", seed=3).fit(X, y).predict_proba(X)
    b = construir_pipeline(prep, estimador, "smote", seed=3).fit(X, y).predict_proba(X)
    c = construir_pipeline(prep, estimador, "smote", seed=4).fit(X, y).predict_proba(X)
    assert np.allclose(a, b) and not np.allclose(a, c)
