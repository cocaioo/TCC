import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gerador_seed import gerar_seed  # noqa: E402
from modelos import (  # noqa: E402
    GRADES, MODELOS, SEM_PESOS, GradientBoostingBalanceado, catalogo, construir_pipeline, n_combinacoes,
)
from preprocessamento import COL_ALVO, construir_preprocessador, dividir, separar_x_y  # noqa: E402

PREVALENCIA = 0.20


@pytest.fixture(scope="module")
def dados():
    treino, _ = dividir(gerar_seed(n=1000, prevalencia=PREVALENCIA, seed=11))
    X, y = separar_x_y(treino)
    return X, y


@pytest.fixture(scope="module")
def amostra(dados):
    """Cerca de 300 linhas do treino (estratificadas), para os ajustes dos testes ficarem rápidos."""
    X, y = dados
    X_s, _, y_s, _ = train_test_split(X, y, train_size=300, stratify=y, random_state=0)
    return X_s, y_s


@pytest.fixture(scope="module")
def prep(dados):
    return construir_preprocessador(dados[0])


def test_catalogo_tem_os_8_modelos_nas_duas_grades():
    for grade in GRADES:
        cat = catalogo(PREVALENCIA, grade)
        assert tuple(cat) == MODELOS and len(cat) == 8
        for estimador, parametros in cat.values():
            assert hasattr(estimador, "fit") and parametros
            assert all(nome.startswith("modelo__") for nome in parametros)


def test_numero_de_combinacoes_das_grades():
    total = {g: sum(n_combinacoes(gr) for _, gr in catalogo(PREVALENCIA, g).values()) for g in GRADES}
    assert total == {"enxuta": 59, "completa": 325}


def test_todo_hiperparametro_de_toda_grade_existe_no_pipeline(prep):
    for grade in GRADES:
        for nome, (estimador, parametros) in catalogo(PREVALENCIA, grade).items():
            combinacoes = list(ParameterGrid(parametros))
            for combinacao in (combinacoes[0], combinacoes[-1]):
                construir_pipeline(prep, estimador).set_params(**combinacao)  # erro de digitação levanta ValueError


def test_cada_pipeline_ajusta_e_devolve_probabilidades(prep, amostra):
    X, y = amostra
    for nome, (estimador, _) in catalogo(PREVALENCIA).items():
        p = construir_pipeline(prep, estimador).fit(X, y).predict_proba(X)
        assert p.shape == (len(X), 2), nome
        assert not np.isnan(p).any(), nome
        assert np.allclose(p.sum(axis=1), 1.0), nome


def test_clone_dos_pipelines_funciona(prep):
    for modo in ("nenhum", "class_weight"):
        for nome, (estimador, _) in catalogo(PREVALENCIA, desbalanceamento=modo).items():
            pipe = construir_pipeline(prep, estimador)
            copia = clone(pipe)
            assert type(copia["modelo"]) is type(pipe["modelo"]), nome
            assert copia["modelo"].get_params() == pipe["modelo"].get_params(), nome


def test_class_weight_nos_modelos_que_aceitam():
    cat = catalogo(PREVALENCIA, desbalanceamento="class_weight")
    base = catalogo(PREVALENCIA, desbalanceamento="nenhum")
    for nome in ("RL", "DT", "RF"):
        assert cat[nome][0].get_params()["class_weight"] == "balanced"
    assert cat["XGBoost"][0].get_params()["scale_pos_weight"] == pytest.approx(4.0)
    assert isinstance(cat["GB"][0], GradientBoostingBalanceado)
    for nome in SEM_PESOS:  # sem suporte no scikit-learn: ficam como no modo "nenhum"
        assert cat[nome][0].get_params() == base[nome][0].get_params()


def test_modo_nenhum_nao_pesa_nada():
    cat = catalogo(PREVALENCIA, desbalanceamento="nenhum")
    for nome in ("RL", "DT", "RF"):
        assert cat[nome][0].get_params()["class_weight"] is None
    assert cat["XGBoost"][0].get_params()["scale_pos_weight"] is None
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


def test_opcoes_invalidas_levantam_erro():
    with pytest.raises(ValueError, match="grade"):
        catalogo(PREVALENCIA, grade="gigante")
    with pytest.raises(ValueError, match="desbalanceamento"):
        catalogo(PREVALENCIA, desbalanceamento="smote")  # entra só no passo 3
    with pytest.raises(ValueError, match="prevalencia"):
        catalogo(0.0, desbalanceamento="class_weight")


def test_preprocessador_e_o_primeiro_passo_e_nao_esta_ajustado(prep, amostra):
    X, y = amostra
    cat = catalogo(PREVALENCIA)
    p1 = construir_pipeline(prep, cat["RL"][0])
    p2 = construir_pipeline(prep, cat["DT"][0])
    assert p1.steps[0][0] == "prep"
    p1.fit(X, y)
    check_is_fitted(p1["prep"]["colunas"])  # controle: o pipeline ajustado aprendeu
    # Ajustar um pipeline não pode contaminar os outros nem o original. Olha-se o ColumnTransformer ("colunas"), que é
    # quem aprende (medianas, escalas): o Pipeline inteiro sempre se declara ajustado, pois o último passo não tem estado.
    for objeto in (prep["colunas"], p2["prep"]["colunas"]):
        with pytest.raises(NotFittedError):
            check_is_fitted(objeto)


def test_mesma_semente_mesmas_probabilidades(prep, amostra):
    X, y = amostra
    for nome in MODELOS:
        a = construir_pipeline(prep, catalogo(PREVALENCIA, seed=7)[nome][0]).fit(X, y).predict_proba(X)
        b = construir_pipeline(prep, catalogo(PREVALENCIA, seed=7)[nome][0]).fit(X, y).predict_proba(X)
        assert np.allclose(a, b), nome


def test_alvo_nao_entra_nas_variaveis(dados):
    X, _ = dados
    assert COL_ALVO not in X.columns


def test_valores_reais_das_grades_ajustam_primeira_do_meio_e_ultima(prep, amostra):
    """`set_params` não valida valores: só ajustar pega `n_neighbors=0`, `weights` inexistente, `max_features` inválido."""
    X, y = amostra
    X, y = X.iloc[:150], y.iloc[:150]
    for grade in GRADES:
        for nome, (estimador, parametros) in catalogo(PREVALENCIA, grade, "class_weight").items():
            combinacoes = list(ParameterGrid(parametros))
            for combinacao in (combinacoes[0], combinacoes[len(combinacoes) // 2], combinacoes[-1]):
                pipe = construir_pipeline(prep, estimador).set_params(**combinacao).fit(X, y)
                assert np.isfinite(pipe.predict_proba(X)).all(), (grade, nome, combinacao)


def test_semente_chega_a_todo_modelo_estocastico():
    for semente in (1, 42):
        cat = catalogo(PREVALENCIA, seed=semente)
        for nome in ("RL", "DT", "RF", "GB", "XGBoost", "MLP"):
            assert cat[nome][0].get_params()["random_state"] == semente, nome


def test_parametros_fixos_decididos_no_plano():
    cat = catalogo(PREVALENCIA)
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


def test_prevalencia_igual_a_1_tambem_e_rejeitada():
    with pytest.raises(ValueError, match="prevalencia"):
        catalogo(1.0, desbalanceamento="class_weight")


def test_grades_devolvidas_nao_compartilham_listas():
    catalogo(PREVALENCIA)["RL"][1]["modelo__C"].append(999)
    assert 999 not in catalogo(PREVALENCIA)["RL"][1]["modelo__C"]
