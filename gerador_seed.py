"""Gerador de dados sintéticos (seed) para o pipeline de sepse neonatal tardia.

Gera uma tabela plana (1 linha por recém-nascido) com as variáveis dos Quadros 2 e 3 do projeto
(págs. 39-40), com desfecho `sepse_tardia` (0/1). Serve apenas para testar o pipeline: as métricas
obtidas com este CSV NÃO têm valor científico.

Ordem de geração:
  1. exposições (parto e internação), com dependências entre si (ex.: prematuridade -> peso, cateteres);
  2. desfecho: modelo logístico com os PESOS_RISCO + ruído; os `n * prevalencia` maiores escores são sépticos;
  3. manifestações (sinais vitais, exames, microbiologia) deslocadas nos sépticos conforme EFEITO_DP;
  4. valores ausentes (só em exames/microbiologia) e variáveis derivadas (IT, RNL, qSOFA, NEWS2).

Todos os números das tabelas abaixo são PALPITES clínicos (a revisar), não estimados de dados reais.

Uso:
    python gerador_seed.py --n 1000 --prevalencia 0.20 --sinal 1.0 --ausentes 1.0 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Tabelas de parâmetros (a revisar)
# --------------------------------------------------------------------------------------

# Pesos do logit do desfecho (log do odds ratio). Palpites de ordem de grandeza.
PESOS_RISCO = {
    "prematuridade_extrema": 0.9,  # IG < 28 semanas
    "cvc": 0.7,
    "npt": 0.5,
    "vm_invasiva": 0.4,
    "ecn": 0.6,
    "atb_previo": 0.4,
    "cateter_umbilical": 0.3,
    "dias_utin_por_10": 0.3,
}

# Variáveis contínuas: (escala, centro, dp, mínimo, máximo).
# escala "log": centro é a mediana e dp é o sigma na escala log; escala "normal": média e dp diretos.
PARAMETROS = {
    "leucocitos": ("log", 12000, 0.35, 500, 80000),
    "frac_seg": ("normal", 0.45, 0.10, 0.02, 0.90),
    "frac_bast": ("normal", 0.04, 0.03, 0.0, 0.40),
    "frac_linf": ("normal", 0.35, 0.09, 0.05, 0.60),
    "plaquetas": ("log", 230000, 0.35, 5000, 800000),
    "pcr": ("log", 3.0, 0.9, 0.1, 400),
    "pct": ("log", 0.5, 1.0, 0.02, 200),
    "glicemia": ("normal", 85, 20, 20, 400),
    "creatinina": ("log", 0.6, 0.3, 0.2, 4.0),
    "bilirrubina_direta": ("log", 0.4, 0.5, 0.1, 10),
    "bilirrubina_indireta": ("log", 5.0, 0.5, 0.2, 25),
    "ph": ("normal", 7.35, 0.06, 6.9, 7.55),
    "lactato": ("log", 1.8, 0.45, 0.5, 15),
    "hemoglobina": ("normal", 14.0, 2.2, 6, 22),
    "sodio": ("normal", 138, 3.5, 120, 160),
    "potassio": ("normal", 4.6, 0.6, 2.5, 7.5),
    "calcio_ionico": ("normal", 1.25, 0.10, 0.7, 1.7),
    "tgo": ("log", 40, 0.5, 5, 1000),
    "tgp": ("log", 20, 0.5, 3, 800),
    # Sinais vitais da última evolução (PAS tem centro dependente da IG, ver _manifestacoes)
    "temp_media": ("normal", 36.8, 0.3, 35.0, 40.0),
    "fc": ("normal", 150, 12, 80, 230),
    "fr": ("normal", 48, 9, 20, 100),
    "spo2": ("normal", 95, 2.5, 70, 100),
    "pas": ("normal", 65, 8, 30, 110),
    "glasgow_neonatal": ("normal", 14.6, 0.8, 3, 15),
}

# Deslocamento nos sépticos, em desvios-padrão (na escala da variável), quando sinal = 1.
# AUC univariada esperada = Phi(|d| * sinal / sqrt(2)).
EFEITO_DP = {
    "leucocitos": 0.6, "frac_seg": -0.2, "frac_bast": 0.8, "frac_linf": -0.4,
    "plaquetas": -0.9, "pcr": 1.4, "pct": 1.5, "glicemia": 0.4, "creatinina": 0.5,
    "bilirrubina_direta": 0.4, "bilirrubina_indireta": 0.1, "ph": -0.8, "lactato": 1.0,
    "hemoglobina": -0.3, "sodio": -0.4, "potassio": 0.2, "calcio_ionico": -0.4,
    "tgo": 0.4, "tgp": 0.3,
    "temp_media": 0.8, "fc": 0.9, "fr": 0.7, "spo2": -0.8, "pas": -0.8, "glasgow_neonatal": -1.2,
}

# Fração do deslocamento que continua na medida seguinte (até 24 h) e ruído serial relativo ao dp.
EVOLUCAO_POS24H = 0.3
RUIDO_SERIAL = 0.4

# Flags clínicas que dependem do desfecho: (prob. nos não sépticos, prob. nos sépticos com sinal = 1).
PROB_FLAG_SEPSE = {
    "vasopressor": (0.03, 0.30),
    "criterios_disfuncao_organica": (0.05, 0.45),
    "historia_sugestiva_infeccao": (0.10, 0.60),
    "hemocultura_positiva": (0.02, 0.40),  # ~40% dos sépticos: A41.9 inclui sepse clínica (cultura negativa)
}

# Taxa de ausência (MCAR) de cada exame na medida `_pre`; a medida `_pos24h` soma ACRESCIMO_POS24H.
TAXA_AUSENTE = {
    "pct": 0.45, "lactato": 0.15, "ph": 0.12, "calcio_ionico": 0.12, "tgo": 0.12, "tgp": 0.12,
    "pcr": 0.08, "bilirrubina_total": 0.08, "bilirrubina_direta": 0.08, "creatinina": 0.05,
    "sodio": 0.05, "potassio": 0.05, "neutrofilos_seg": 0.05, "bastoes": 0.05, "linfocitos": 0.05,
    "leucocitos": 0.02, "plaquetas": 0.02,
}
TAXA_AUSENTE_PADRAO = 0.03
ACRESCIMO_POS24H = 0.10
TAXA_AUSENTE_HEMOCULTURA = 0.03

# Casas decimais de cada exame (arredondado antes de injetar ausentes e calcular derivadas).
CASAS_DECIMAIS = {
    "leucocitos": 0, "neutrofilos_seg": 0, "bastoes": 0, "linfocitos": 0, "plaquetas": 0,
    "pcr": 1, "pct": 2, "glicemia": 0, "creatinina": 2, "bilirrubina_direta": 1,
    "bilirrubina_total": 1, "ph": 2, "lactato": 1, "hemoglobina": 1, "hematocrito": 0,
    "sodio": 0, "potassio": 1, "calcio_ionico": 2, "tgo": 0, "tgp": 0,
}

# Variáveis cujo valor pode ser CONSEQUÊNCIA da sepse no dado real (vazamento de alvo). No seed apenas
# causam o desfecho (ou dependem dele), mas devem ser discutidas na etapa de pré-processamento:
#  - totais da internação: a sepse prolonga a internação e o uso de dispositivos;
#  - dias_ate_dx_sepse: só existe para quem teve sepse;
#  - hemocultura_*: resultado diretamente ligado ao diagnóstico.
VARS_RISCO_VAZAMENTO = [
    "dias_utin", "vm_dias", "cvc_dias", "npt_dias", "dias_ate_dx_sepse",
    "hemocultura_resultado", "hemocultura_agente", "perfil_sensibilidade",
]

# Exames seriados (nome base do dicionário PARAMETROS ou derivado), na ordem das colunas do CSV.
EXAMES = [
    "leucocitos", "neutrofilos_seg", "bastoes", "relacao_it", "plaquetas", "pcr", "pct", "glicemia",
    "creatinina", "bilirrubina_total", "bilirrubina_direta", "bilirrubina_indireta", "ph", "lactato",
    "hemoglobina", "hematocrito", "sodio", "potassio", "calcio_ionico", "tgo", "tgp",
    "relacao_neutrofilos_linfocitos",
]

CLASSES_ATB = {  # probabilidade de cada classe, dado atb_previo = 1
    "penicilinas": 0.70, "cefalosporinas": 0.40, "aminoglicosideos": 0.65,
    "carbapenemicos": 0.10, "glicopeptideos": 0.15,
}

# --------------------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------------------


def _sortear_flag(rng, n, p):
    """Sorteia n valores 0/1 com probabilidade `p` (escalar ou vetor de tamanho n)."""
    return (rng.random(n) < p).astype(int)


def _prob_por_sepse(sepse, nome, sinal):
    """Probabilidade da flag `nome`: vai de p_base (sem sepse) até p_sepse (com sepse), escalada por `sinal`."""
    base, com_sepse = PROB_FLAG_SEPSE[nome]
    return np.where(sepse == 1, base + (com_sepse - base) * sinal, base)


def _para_escala(nome, valores):
    return np.log(valores) if PARAMETROS[nome][0] == "log" else valores


def _da_escala(nome, valores):
    _, _, _, minimo, maximo = PARAMETROS[nome]
    if PARAMETROS[nome][0] == "log":
        valores = np.exp(valores)
    return np.clip(valores, minimo, maximo)


def _medida(rng, nome, sepse, sinal, centro=None):
    """Valor da variável `nome` na escala de modelagem (log ou linear), deslocado nos sépticos."""
    escala, centro_padrao, dp, _, _ = PARAMETROS[nome]
    centro = centro_padrao if centro is None else centro
    base = np.log(centro) if escala == "log" else centro
    return base + EFEITO_DP[nome] * sinal * sepse * dp + dp * rng.normal(size=len(sepse))


def _medida_seguinte(rng, nome, anterior, sepse, sinal):
    """Segunda medida (até 24 h após a suspeita): continua a evolução da primeira, com ruído serial."""
    dp = PARAMETROS[nome][2]
    desloc = EVOLUCAO_POS24H * EFEITO_DP[nome] * sinal * sepse * dp
    return anterior + desloc + RUIDO_SERIAL * dp * rng.normal(size=len(sepse))


def _aplicar_ausentes(rng, valores, taxa):
    """Troca por NaN (MCAR) uma fração `taxa` dos valores."""
    valores = np.asarray(valores, dtype=float).copy()
    valores[rng.random(len(valores)) < taxa] = np.nan
    return valores


# --------------------------------------------------------------------------------------
# Blocos de geração
# --------------------------------------------------------------------------------------


def _demograficas_e_parto(rng, n):
    ig = np.clip(np.round(rng.normal(32.5, 4.2, n)), 22, 41).astype(int)
    peso_medio = np.interp(
        ig, [22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 41],
        [500, 650, 850, 1150, 1450, 1900, 2350, 2850, 3200, 3450, 3550],
    )
    peso = np.clip(np.round(peso_medio * (1 + 0.12 * rng.normal(size=n))), 400, None).astype(int)

    # Apgar: menor nos mais prematuros; o de 5º minuto costuma ser >= ao de 1º.
    base_apgar = 8 - 0.25 * np.maximum(0, 34 - ig)
    apgar1 = np.clip(np.round(rng.normal(base_apgar, 1.5)), 0, 10).astype(int)
    apgar5 = np.clip(apgar1 + np.maximum(0, np.round(rng.normal(1.3, 1.2, n))), 0, 10).astype(int)

    raca = np.array(["parda", "branca", "preta", "amarela", "indigena"])
    pesos_raca = [0.62, 0.25, 0.10, 0.02, 0.01]
    raca_rn = rng.choice(raca, n, p=pesos_raca)
    raca_mae = np.where(rng.random(n) < 0.9, raca_rn, rng.choice(raca, n, p=pesos_raca))

    rpmo = _sortear_flag(rng, n, 0.10 + 0.15 * (ig < 34))
    corioamnionite = _sortear_flag(rng, n, 0.03 + 0.12 * rpmo)
    itu_materna = _sortear_flag(rng, n, 0.10)
    pre_eclampsia = _sortear_flag(rng, n, 0.08 + 0.12 * (ig < 34))
    p_atb_materno = np.clip(0.10 + 0.40 * np.maximum(corioamnionite, rpmo) + 0.10 * itu_materna, 0, 1)

    dados = {
        "sexo": rng.choice(["masculino", "feminino"], n, p=[0.53, 0.47]),
        "idade_gestacional_sem": ig,
        "peso_nascer_g": peso,
        "apgar_1min": apgar1,
        "apgar_5min": apgar5,
        "tipo_parto": rng.choice(["cesariana", "vaginal"], n, p=[0.60, 0.40]),
        "idade_materna": np.clip(np.round(rng.normal(26, 6.5, n)), 14, 45).astype(int),
        "raca_cor_rn": raca_rn,
        "raca_cor_mae": raca_mae,
        "n_gestacoes_anteriores": np.clip(rng.poisson(1.1, n), 0, 8),
        "itu_materna": itu_materna,
        "rpmo": rpmo,
        "corioamnionite": corioamnionite,
        "pre_eclampsia": pre_eclampsia,
        "atb_materno_intraparto": _sortear_flag(rng, n, p_atb_materno),
    }
    return dados, ig


def _internacao(rng, n, ig, atb_materno):
    """Exposições da internação; dispositivos e comorbidades ficam mais prováveis com menor IG."""
    imaturidade = np.maximum(0, 34 - ig)

    # Média de dias cresce com a prematuridade; mínimo de 4 dias (critério de inclusão: > 72 h).
    media_dias = 8 + 2.2 * np.maximum(0, 36 - ig)
    dias_utin = np.clip(4 + rng.poisson(media_dias * rng.gamma(4, 0.25, n)), 4, 150)

    def dias_com(flag, fracao, minimo):
        return flag * np.minimum(dias_utin, minimo + rng.poisson(fracao * dias_utin))

    vm = _sortear_flag(rng, n, np.clip(0.15 + 0.06 * imaturidade, 0, 0.9))
    cvc = _sortear_flag(rng, n, np.clip(0.15 + 0.05 * imaturidade, 0, 0.9))
    npt = _sortear_flag(rng, n, np.clip(0.10 + 0.05 * imaturidade, 0, 0.9))
    cateter_umb = _sortear_flag(rng, n, np.clip(0.10 + 0.04 * imaturidade, 0, 0.9))
    surfactante = _sortear_flag(rng, n, np.clip(0.05 + 0.07 * imaturidade, 0, 0.9))
    ecn = _sortear_flag(rng, n, np.clip(0.01 + 0.015 * np.maximum(0, 32 - ig), 0, 0.5))

    # Hemorragia intraventricular só em prematuros (< 32 semanas); graus I-IV.
    tem_hiv = _sortear_flag(rng, n, np.where(ig < 32, 0.05 + 0.025 * (32 - ig), 0))
    hiv_grau = tem_hiv * rng.choice([1, 2, 3, 4], n, p=[0.50, 0.25, 0.15, 0.10])

    atb_previo = _sortear_flag(rng, n, 0.30 + 0.20 * (ig < 32) + 0.10 * atb_materno)
    classes = {
        f"atb_prev_{nome}": atb_previo * _sortear_flag(rng, n, p) for nome, p in CLASSES_ATB.items()
    }
    # Quem usou antibiótico precisa ter ao menos uma classe registrada.
    sem_classe = (atb_previo == 1) & (sum(classes.values()) == 0)
    sorteada = rng.choice(list(classes), n)
    for nome, valores in classes.items():
        valores[sem_classe & (sorteada == nome)] = 1

    dados = {
        "dias_utin": dias_utin,
        "vm_invasiva": vm,
        "vm_dias": dias_com(vm, 0.25, 1),
        "cvc": cvc,
        "cvc_dias": dias_com(cvc, 0.5, 2),
        "cateter_umbilical": cateter_umb,
        "npt": npt,
        "npt_dias": dias_com(npt, 0.4, 2),
        "atb_previo": atb_previo,
        **classes,
        "ecn": ecn,
        "hiv_grau": hiv_grau,
        "surfactante": surfactante,
    }
    return dados


def _desfecho(rng, n, ig, interna, prevalencia):
    """Escore logístico (PESOS_RISCO + ruído logístico); os n*prevalencia maiores escores são sépticos."""
    fatores = {
        "prematuridade_extrema": (ig < 28).astype(float),
        "cvc": interna["cvc"],
        "npt": interna["npt"],
        "vm_invasiva": interna["vm_invasiva"],
        "ecn": interna["ecn"],
        "atb_previo": interna["atb_previo"],
        "cateter_umbilical": interna["cateter_umbilical"],
        "dias_utin_por_10": interna["dias_utin"] / 10,
    }
    escore = sum(PESOS_RISCO[nome] * valores for nome, valores in fatores.items())
    escore = escore + rng.logistic(size=n)
    sepse = np.zeros(n, dtype=int)
    sepse[np.argsort(-escore)[: int(round(n * prevalencia))]] = 1
    return sepse


def _vitais_e_flags(rng, n, ig, sepse, sinal):
    """Sinais vitais, flags dependentes do desfecho e Glasgow (última evolução)."""
    pas_centro = 28 + 1.1 * ig  # pressão arterial média sobe com a idade gestacional
    temp_media = _da_escala("temp_media", _medida(rng, "temp_media", sepse, sinal))
    temp_max = np.minimum(temp_media + np.abs(rng.normal(0.4, 0.2, n)), 41.0)
    temp_min = np.maximum(temp_media - np.abs(rng.normal(0.4, 0.2, n)), 34.0)
    pas = _da_escala("pas", _medida(rng, "pas", sepse, sinal, centro=pas_centro))
    pad = np.clip(0.6 * pas + rng.normal(0, 4, n), 20, pas - 8)

    vasopressor = _sortear_flag(rng, n, _prob_por_sepse(sepse, "vasopressor", sinal))
    disfuncao = _sortear_flag(rng, n, _prob_por_sepse(sepse, "criterios_disfuncao_organica", sinal))
    lam_apneia = (0.5 + 0.15 * np.maximum(0, 34 - ig)) * np.exp(0.8 * sinal * sepse)

    return {
        "historia_sugestiva_infeccao": _sortear_flag(
            rng, n, _prob_por_sepse(sepse, "historia_sugestiva_infeccao", sinal)),
        "criterios_disfuncao_organica": np.maximum(disfuncao, vasopressor),  # vasopressor implica disfunção
        "temp_min": np.round(temp_min, 1),
        "temp_max": np.round(temp_max, 1),
        "temp_media": np.round(temp_media, 1),
        "fc": np.round(_da_escala("fc", _medida(rng, "fc", sepse, sinal))).astype(int),
        "fr": np.round(_da_escala("fr", _medida(rng, "fr", sepse, sinal))).astype(int),
        "spo2": np.round(_da_escala("spo2", _medida(rng, "spo2", sepse, sinal))).astype(int),
        "pas": np.round(pas).astype(int),
        "pad": np.round(pad).astype(int),
        "vasopressor": vasopressor,
        "apneia_bradicardia_24h": rng.poisson(lam_apneia),
        "glasgow_neonatal": np.round(
            _da_escala("glasgow_neonatal", _medida(rng, "glasgow_neonatal", sepse, sinal))).astype(int),
    }


def _exames(rng, n, sepse, sinal, ausentes):
    """Exames seriados (`_pre` e `_pos24h`), com ausentes; relações e bilirrubina indireta são derivadas."""
    brutos = {}  # nome -> [pre, pos] na escala original, sem arredondar
    for nome in PARAMETROS:
        if nome in EFEITO_DP and nome not in (
                "temp_media", "fc", "fr", "spo2", "pas", "glasgow_neonatal"):
            pre = _medida(rng, nome, sepse, sinal)
            pos = _medida_seguinte(rng, nome, pre, sepse, sinal)
            brutos[nome] = [_da_escala(nome, pre), _da_escala(nome, pos)]

    # Hematócrito ~ 3 x hemoglobina; leucograma = leucócitos x frações (soma limitada a 95%).
    # Bilirrubina total = direta + indireta, medida separadamente; a indireta é derivada depois.
    medidas = {}
    for i, sufixo in enumerate(("pre", "pos24h")):
        v = {nome: serie[i] for nome, serie in brutos.items()}
        frac_bast, frac_linf = v["frac_bast"], v["frac_linf"]
        frac_seg = np.minimum(v["frac_seg"], 0.95 - frac_bast - frac_linf)
        frac_seg = np.maximum(frac_seg, 0.02)
        leuc = v["leucocitos"]
        medidas[sufixo] = {
            "leucocitos": leuc,
            "neutrofilos_seg": leuc * frac_seg,
            "bastoes": leuc * frac_bast,
            "linfocitos": leuc * frac_linf,
            "hematocrito": np.clip(3 * v["hemoglobina"] + rng.normal(0, 1.5, n), 20, 65),
            "bilirrubina_total": np.round(v["bilirrubina_direta"], 1) + np.round(v["bilirrubina_indireta"], 1),
            **{k: v[k] for k in (
                "plaquetas", "pcr", "pct", "glicemia", "creatinina", "bilirrubina_direta", "ph", "lactato",
                "hemoglobina", "sodio", "potassio", "calcio_ionico", "tgo", "tgp")},
        }

    colunas = {}
    for sufixo, v in medidas.items():
        acrescimo = ACRESCIMO_POS24H if sufixo == "pos24h" else 0.0
        for nome, valores in v.items():
            taxa = min(0.95, (TAXA_AUSENTE.get(nome, TAXA_AUSENTE_PADRAO) + acrescimo) * ausentes)
            arredondado = np.round(valores, CASAS_DECIMAIS[nome])
            colunas[f"{nome}_{sufixo}"] = _aplicar_ausentes(rng, arredondado, taxa)

    # Derivadas a partir das colunas já com NaN: o NaN se propaga como no conector real.
    for sufixo in ("pre", "pos24h"):
        seg, bast = colunas[f"neutrofilos_seg_{sufixo}"], colunas[f"bastoes_{sufixo}"]
        linf = colunas.pop(f"linfocitos_{sufixo}")
        # Imaturos/totais aproximado só com bastões (não há metamielócitos no quadro de variáveis).
        colunas[f"relacao_it_{sufixo}"] = np.round(bast / (bast + seg), 2)
        colunas[f"relacao_neutrofilos_linfocitos_{sufixo}"] = np.round((seg + bast) / linf, 2)
        colunas[f"bilirrubina_indireta_{sufixo}"] = np.round(
            colunas[f"bilirrubina_total_{sufixo}"] - colunas[f"bilirrubina_direta_{sufixo}"], 1)

    return {f"{nome}_{sufixo}": colunas[f"{nome}_{sufixo}"] for nome in EXAMES for sufixo in ("pre", "pos24h")}


def _microbiologia(rng, n, sepse, sinal, interna, ausentes):
    positiva = _sortear_flag(rng, n, _prob_por_sepse(sepse, "hemocultura_positiva", sinal)) == 1
    agentes = ["scn", "staphylococcus_aureus", "klebsiella", "e_coli", "candida", "enterococcus",
               "pseudomonas", "outros"]
    agente = rng.choice(agentes, n, p=[0.35, 0.12, 0.18, 0.06, 0.10, 0.06, 0.05, 0.08])
    sensibilidade = rng.choice(["sensivel", "resistente_oxacilina", "multirresistente"], n, p=[0.5, 0.3, 0.2])

    resultado = np.where(positiva, "positiva", "negativa").astype(object)
    agente = np.where(positiva, agente, None).astype(object)
    sensibilidade = np.where(positiva, sensibilidade, None).astype(object)
    # Resultado ausente leva junto agente e perfil de sensibilidade.
    sem_resultado = rng.random(n) < min(0.95, TAXA_AUSENTE_HEMOCULTURA * ausentes)
    for coluna in (resultado, agente, sensibilidade):
        coluna[sem_resultado] = None

    cvc, vm = interna["cvc"], interna["vm_invasiva"]
    # Colonização por SCN / Klebsiella é mais provável em quem usa cateter central.
    p_colonizacao = np.where(cvc[:, None] == 1, [[0.55, 0.30, 0.15]], [[0.70, 0.20, 0.10]])
    sorteio = rng.random(n)[:, None] > np.cumsum(p_colonizacao, axis=1)
    colonizacao = np.array(["nenhuma", "scn", "klebsiella"])[sorteio.sum(axis=1).clip(0, 2)]

    return {
        "hemocultura_resultado": resultado,
        "hemocultura_agente": agente,
        "perfil_sensibilidade": sensibilidade,
        "hist_infeccao_ira": _sortear_flag(rng, n, 0.05 + 0.10 * cvc + 0.05 * vm),
        "colonizacao": colonizacao,
    }


def _news2(fr, spo2, pas, fc, temp, glasgow, oxigenio):
    """NEWS2 (escala 1 de SpO2) com os cortes padronizados de adulto, como o projeto manda (p. 43)."""
    pontos_fr = np.select([fr <= 8, fr <= 11, fr <= 20, fr <= 24], [3, 1, 0, 2], 3)
    pontos_spo2 = np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95], [3, 2, 1], 0)
    pontos_pas = np.select([pas <= 90, pas <= 100, pas <= 110, pas <= 219], [3, 2, 1, 0], 3)
    pontos_fc = np.select([fc <= 40, fc <= 50, fc <= 90, fc <= 110, fc <= 130], [3, 1, 0, 1, 2], 3)
    pontos_temp = np.select([temp <= 35.0, temp <= 36.0, temp <= 38.0, temp <= 39.0], [3, 1, 0, 1], 2)
    pontos_consciencia = np.where(glasgow < 15, 3, 0)
    # Sem coluna de O2 suplementar nas variáveis do projeto, a ventilação invasiva faz o papel de proxy.
    return (pontos_fr + pontos_spo2 + pontos_pas + pontos_fc + pontos_temp + pontos_consciencia
            + 2 * oxigenio)


def _qsofa(fr, pas, glasgow):
    """qSOFA: FR >= 22, PAS <= 100 mmHg e alteração do estado mental (Glasgow < 15)."""
    return (fr >= 22).astype(int) + (pas <= 100).astype(int) + (glasgow < 15).astype(int)


# --------------------------------------------------------------------------------------
# API pública e CLI
# --------------------------------------------------------------------------------------


def gerar_seed(n=1000, prevalencia=0.20, sinal=1.0, ausentes=1.0, seed=42):
    """Gera o DataFrame sintético (1 linha por recém-nascido).

    sinal:     0 = classes indistinguíveis; 1 = efeitos de EFEITO_DP por inteiro.
    ausentes:  multiplicador das taxas de TAXA_AUSENTE (0 = sem ausentes).
    """
    if not 0 < prevalencia < 1:
        raise ValueError("prevalencia deve estar entre 0 e 1")
    if not 0 <= sinal <= 1:
        raise ValueError("sinal deve estar entre 0 e 1")
    rng = np.random.default_rng(seed)

    demograficas, ig = _demograficas_e_parto(rng, n)
    interna = _internacao(rng, n, ig, demograficas["atb_materno_intraparto"])
    sepse = _desfecho(rng, n, ig, interna, prevalencia)
    vitais = _vitais_e_flags(rng, n, ig, sepse, sinal)
    exames = _exames(rng, n, sepse, sinal, ausentes)
    micro = _microbiologia(rng, n, sepse, sinal, interna, ausentes)

    # Intervalo até o diagnóstico: só existe para sépticos, entre o 3º dia de vida e o fim da internação.
    dias_dx = np.where(sepse == 1, rng.integers(3, interna["dias_utin"] + 1), np.nan)
    escores = {
        "qsofa": _qsofa(vitais["fr"], vitais["pas"], vitais["glasgow_neonatal"]),
        "news2": _news2(vitais["fr"], vitais["spo2"], vitais["pas"], vitais["fc"], vitais["temp_media"],
                        vitais["glasgow_neonatal"], interna["vm_invasiva"]),
    }

    # Ordem das colunas segue os Quadros 2 e 3 do projeto.
    clinicas = {**interna, **vitais, "dias_ate_dx_sepse": dias_dx}
    largura = max(4, len(str(n)))
    df = pd.DataFrame({
        "id_paciente": [f"RN{i + 1:0{largura}d}" for i in range(n)],
        **demograficas, **clinicas, **exames, **micro, **escores,
        "sepse_tardia": sepse,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Gera dados sintéticos de sepse neonatal tardia.")
    parser.add_argument("--n", type=int, default=1000, help="número de recém-nascidos (padrão 1000)")
    parser.add_argument("--prevalencia", type=float, default=0.20, help="proporção de sepse tardia")
    parser.add_argument("--sinal", type=float, default=1.0, help="0 = sem sinal, 1 = efeito completo")
    parser.add_argument("--ausentes", type=float, default=1.0, help="multiplicador das taxas de ausência")
    parser.add_argument("--seed", type=int, default=42, help="semente do gerador aleatório")
    parser.add_argument("--saida", type=Path, default=Path("data/seed_sepse.csv"))
    args = parser.parse_args()

    df = gerar_seed(args.n, args.prevalencia, args.sinal, args.ausentes, args.seed)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False, encoding="utf-8")
    print(f"{len(df)} linhas x {df.shape[1]} colunas -> {args.saida} "
          f"(sepse tardia: {df['sepse_tardia'].mean():.1%})")


if __name__ == "__main__":
    main()
