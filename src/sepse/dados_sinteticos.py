"""Gerador de pacientes fictícios para testar o pipeline.

Os dados NÃO servem como resultado do TCC: só validam o código. O desfecho é
gerado a partir de um sinal conhecido (SINAL_PLANTADO), para conferirmos se os
modelos e o SHAP o recuperam.

Geração em 3 etapas:
  A. características de base (antes da sepse), coerentes entre si;
  B. risco de sepse = sigmoide(b0 + soma dos efeitos de SINAL_PLANTADO) -> desfecho;
  C. exames, sinais vitais e escores condicionados ao desfecho (consequência da infecção).

Uso:
  python -m sepse.dados_sinteticos --n 700 --prevalencia 0.2 --faltantes 0.1 --semente 42
"""

import argparse
import sys

import numpy as np
import pandas as pd

from sepse import schema
from sepse.io import aplicar_tipos, salvar_csv

# Coeficientes (escala logit) dos fatores de risco plantados na etapa B.
# Variáveis fora desta lista e fora da etapa C (sexo, raça, tipo de parto,
# idade materna, sódio...) são ruído puro.
SINAL_PLANTADO = {
    "semanas_abaixo_de_37": 0.12,  # por semana de prematuridade
    "peso_abaixo_de_1500_por_500g": 0.30,
    "cvc": 0.60,
    "cvc_dias": 0.04,  # por dia de cateter
    "vmi": 0.50,
    "npt": 0.40,
    "colonizacao_scn": 0.50,
    "colonizacao_klebsiella": 0.70,
}

# Taxa própria de faltantes da procalcitonina ("quando disponível", Quadro 3).
FALTANTES_PCT = 0.60

# Peso mediano (g) por idade gestacional — curva aproximada.
_CURVA_PESO = ([22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
               [500, 650, 850, 1100, 1450, 1800, 2250, 2700, 3100, 3400, 3550])


def _sigmoide(x):
    return 1 / (1 + np.exp(-x))


def _bernoulli(rng, p):
    return (rng.random(np.shape(p)) < p).astype(int)


def _lognormal(rng, mediana, sigma, n):
    return mediana * np.exp(rng.normal(0, sigma, n))


def _escolher(rng, valores, n, p=None):
    return rng.choice(np.array(valores, dtype=object), size=n, p=p)


def _etapa_a_base(rng, n):
    d = {}
    ig = np.floor(22 + 20 * rng.beta(2.5, 1.8, n)).astype(int)  # UTIN: mais prematuros
    d["idade_gestacional_sem"] = ig
    d["peso_nascer_g"] = np.interp(ig, *_CURVA_PESO) * _lognormal(rng, 1, 0.15, n)

    prem = _sigmoide((32 - ig) / 2)  # 0 = termo, 1 = prematuro extremo
    apgar5 = np.round(9 - 0.2 * np.clip(37 - ig, 0, None) + rng.normal(0, 1.2, n))
    d["apgar_5min"] = np.clip(apgar5, 0, 10)
    d["apgar_1min"] = np.clip(d["apgar_5min"] - rng.poisson(1.5, n), 0, 10)

    d["sexo"] = _escolher(rng, ["M", "F"], n)
    d["tipo_parto"] = _escolher(rng, ["vaginal", "cesariana"], n, [0.45, 0.55])
    d["idade_materna"] = np.round(rng.normal(26, 6.5, n))
    pesos_raca = [0.20, 0.10, 0.67, 0.02, 0.01]
    d["raca_cor_mae"] = _escolher(rng, schema.RACAS, n, pesos_raca)
    d["raca_cor_rn"] = np.where(rng.random(n) < 0.85, d["raca_cor_mae"],
                                _escolher(rng, schema.RACAS, n, pesos_raca))
    d["n_gestacoes_anteriores"] = rng.poisson(1.2, n)
    d["itu_materna"] = _bernoulli(rng, 0.15)
    d["rotura_prematura_membranas"] = _bernoulli(rng, 0.10 + 0.20 * prem)
    d["corioamnionite"] = _bernoulli(rng, 0.03 + 0.07 * prem)
    d["pre_eclampsia"] = _bernoulli(rng, 0.12 + 0.10 * prem)
    d["atb_materno_intraparto"] = _bernoulli(rng, 0.20 + 0.20 * prem)

    # Dispositivos/terapias: mais prováveis em prematuros; dias = 0 quando não usou.
    for nome, p_base, p_prem, dias_base, dias_prem in [
        ("vmi", 0.10, 0.70, 3, 15),
        ("cvc", 0.15, 0.60, 5, 12),
        ("npt", 0.15, 0.70, 5, 10),
    ]:
        uso = _bernoulli(rng, p_base + p_prem * prem)
        d[nome] = uso
        d[f"{nome}_dias"] = uso * (1 + rng.poisson(dias_base + dias_prem * prem))
    d["cateter_umbilical"] = _bernoulli(rng, 0.10 + 0.60 * prem)
    d["surfactante"] = _bernoulli(rng, 0.05 + 0.80 * prem)
    d["hiv_grau"] = np.where(rng.random(n) < 0.05 + 0.30 * prem,
                             rng.integers(1, 5, n), 0)

    d["colonizacao_scn"] = _bernoulli(rng, 0.10 + 0.20 * prem)
    d["colonizacao_klebsiella"] = _bernoulli(rng, 0.05 + 0.15 * prem)
    return d, prem


def _logit_risco(d):
    s = SINAL_PLANTADO
    return (
        s["semanas_abaixo_de_37"] * np.clip(37 - d["idade_gestacional_sem"], 0, None)
        + s["peso_abaixo_de_1500_por_500g"] * np.clip((1500 - d["peso_nascer_g"]) / 500, 0, None)
        + s["cvc"] * d["cvc"]
        + s["cvc_dias"] * d["cvc_dias"]
        + s["vmi"] * d["vmi"]
        + s["npt"] * d["npt"]
        + s["colonizacao_scn"] * d["colonizacao_scn"]
        + s["colonizacao_klebsiella"] * d["colonizacao_klebsiella"]
    )


def _calibrar_intercepto(logit, prevalencia):
    """Busca binária de b0 para que a média do risco seja a prevalência pedida."""
    baixo, alto = -20.0, 20.0
    for _ in range(60):
        b0 = (baixo + alto) / 2
        if _sigmoide(b0 + logit).mean() < prevalencia:
            baixo = b0
        else:
            alto = b0
    return b0


def _etapa_c_consequencias(rng, d, prem, sepse):
    n = len(sepse)
    s = sepse.astype(bool)

    def por_grupo(sem, com):
        return np.where(s, com, sem)

    # Clínicas que acompanham a infecção
    d["historia_infeccao"] = _bernoulli(rng, por_grupo(0.10, 0.50))
    d["disfuncao_organica"] = _bernoulli(rng, por_grupo(0.05, 0.40))
    d["enterocolite_necrosante"] = _bernoulli(rng, por_grupo(0.04, 0.15) + 0.04 * prem)
    d["vasopressor"] = _bernoulli(rng, por_grupo(0.05, 0.30))
    d["vasopressor_droga"] = np.where(d["vasopressor"] == 1,
                                      _escolher(rng, schema.VASOPRESSORES, n), None)
    d["atb_previo"] = _bernoulli(rng, por_grupo(0.30, 0.50))
    d["atb_previo_classes"] = np.where(d["atb_previo"] == 1,
                                       _escolher(rng, schema.CLASSES_ATB, n, [0.5, 0.25, 0.15, 0.10]),
                                       None)
    d["apneia_bradicardia_24h"] = rng.poisson(por_grupo(1.0, 3.0) + 2 * prem)
    d["historico_iras"] = _bernoulli(rng, por_grupo(0.05, 0.20))

    # Sinais vitais
    d["temp_media"] = rng.normal(por_grupo(36.8, 37.1), 0.3)
    d["temp_min"] = d["temp_media"] - np.abs(rng.normal(por_grupo(0.4, 0.8), 0.2))
    d["temp_max"] = d["temp_media"] + np.abs(rng.normal(por_grupo(0.4, 0.9), 0.2))
    d["fc"] = rng.normal(por_grupo(145, 162), 15)
    d["fr"] = rng.normal(por_grupo(48, 58), 10)
    d["spo2"] = np.clip(rng.normal(por_grupo(95, 91), 3), None, 100)
    d["pas"] = rng.normal(d["idade_gestacional_sem"] + 18 - 6 * s, 8)
    d["pad"] = d["pas"] * rng.uniform(0.5, 0.7, n)

    # Laboratoriais (distribuições sobrepostas de propósito)
    d["leucocitos"] = _lognormal(rng, por_grupo(12000, 13000), por_grupo(0.30, 0.55), n)
    d["relacao_it"] = rng.beta(por_grupo(2, 4), por_grupo(16, 10))
    neutro_total = d["leucocitos"] * rng.uniform(0.40, 0.65, n)
    d["bastoes"] = neutro_total * d["relacao_it"]
    d["neutrofilos_segmentados"] = neutro_total - d["bastoes"]
    linfocitos = d["leucocitos"] * rng.uniform(0.20, 0.40, n)
    d["relacao_nl"] = neutro_total / linfocitos * por_grupo(1.0, 1.6)
    d["plaquetas"] = rng.normal(por_grupo(240000, 150000), 70000)
    d["pcr"] = _lognormal(rng, por_grupo(3, 25), 0.9, n)
    d["pct"] = _lognormal(rng, por_grupo(0.4, 3.0), 0.9, n)
    d["lactato"] = _lognormal(rng, por_grupo(1.5, 2.8), 0.35, n)
    d["ph"] = rng.normal(por_grupo(7.35, 7.29), 0.05)
    d["glicemia"] = rng.normal(por_grupo(85, 110), 30)
    # Sem efeito do desfecho (ruído)
    d["creatinina"] = _lognormal(rng, 0.6 + 0.3 * prem, 0.3, n)
    d["bilirrubina_direta"] = _lognormal(rng, 0.5, 0.5, n)
    d["bilirrubina_indireta"] = _lognormal(rng, 6.0, 0.5, n)
    d["bilirrubina_total"] = d["bilirrubina_direta"] + d["bilirrubina_indireta"]
    d["hemoglobina"] = rng.normal(15, 2.5, n)
    d["hematocrito"] = d["hemoglobina"] * 3 + rng.normal(0, 1.5, n)
    d["sodio"] = rng.normal(138, 4, n)
    d["potassio"] = rng.normal(4.8, 0.6, n)
    d["calcio_ionizado"] = rng.normal(1.2, 0.12, n)
    d["tgo"] = _lognormal(rng, 35, 0.5, n)
    d["tgp"] = _lognormal(rng, 15, 0.5, n)

    # Escores (por ora sorteados; cálculo real a partir dos componentes vem depois)
    d["glasgow_neonatal"] = 15 - rng.poisson(por_grupo(0.5, 2.5))
    d["news2"] = rng.poisson(por_grupo(2.0, 5.0))
    d["qsofa"] = rng.binomial(3, por_grupo(0.10, 0.40))
    d["nsofa"] = rng.poisson(por_grupo(1.0, 4.0))

    # Colunas com suspeita de vazamento: comportamento "real", para o pipeline excluí-las
    d["dias_utin"] = 3 + rng.poisson(10 + 30 * prem) + s * rng.poisson(12, n)
    d["dias_ate_diagnostico"] = np.where(s, np.minimum(4 + rng.poisson(6, n), d["dias_utin"]), np.nan)
    d["hemocultura_positiva"] = _bernoulli(rng, por_grupo(0.03, 0.60))
    positiva = d["hemocultura_positiva"] == 1
    d["hemocultura_agente"] = np.where(
        positiva, _escolher(rng, schema.AGENTES, n, [0.40, 0.25, 0.15, 0.10, 0.10]), None)
    d["perfil_sensibilidade"] = np.where(
        positiva, _escolher(rng, schema.PERFIS_SENSIBILIDADE, n, [0.65, 0.35]), None)


def _limitar_faixas(df):
    for c in schema.COLUNAS:
        if c.tipo != schema.CATEGORICA:
            df[c.nome] = df[c.nome].clip(c.minimo, c.maximo)
            if c.tipo == schema.NUMERICA:
                df[c.nome] = df[c.nome].round(4)
            else:
                df[c.nome] = df[c.nome].round()
    return df


def _inserir_faltantes(rng, df, perc_faltantes):
    """Apaga valores ao acaso nos preditores (nunca no id nem no alvo)."""
    for nome in schema.PREDITORES:
        taxa = FALTANTES_PCT if nome == "pct" else perc_faltantes
        df.loc[rng.random(len(df)) < taxa, nome] = None
    return df


def gerar_dados(
    n_pacientes: int = 700,
    prevalencia: float = 0.20,
    perc_faltantes: float = 0.10,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera um DataFrame de pacientes fictícios no formato do schema."""
    rng = np.random.default_rng(semente)

    d, prem = _etapa_a_base(rng, n_pacientes)
    logit = _logit_risco(d)
    risco = _sigmoide(_calibrar_intercepto(logit, prevalencia) + logit)
    sepse = _bernoulli(rng, risco)
    _etapa_c_consequencias(rng, d, prem, sepse)

    df = pd.DataFrame(d)
    df[schema.ID] = np.arange(1, n_pacientes + 1)
    df[schema.ALVO] = sepse
    df = _limitar_faixas(df.astype({c.nome: "object" for c in schema.COLUNAS
                                    if c.tipo == schema.CATEGORICA}))
    df = _inserir_faltantes(rng, df, perc_faltantes)
    return aplicar_tipos(df)


def main(argv=None):
    p = argparse.ArgumentParser(description="Gera dados sintéticos de sepse neonatal tardia.")
    p.add_argument("--n", type=int, default=700, help="número de pacientes")
    p.add_argument("--prevalencia", type=float, default=0.20)
    p.add_argument("--faltantes", type=float, default=0.10, help="fração de valores ausentes")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--saida", default="data/sintetico/sintetico.csv")
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # acentos no console do Windows

    df = gerar_dados(args.n, args.prevalencia, args.faltantes, args.semente)
    salvar_csv(df, args.saida)

    alvo = df[schema.ALVO]
    print(f"{len(df)} pacientes salvos em {args.saida}")
    print(f"Prevalência de sepse tardia: {alvo.mean():.1%}")
    print(df.groupby(schema.ALVO)[["idade_gestacional_sem", "cvc", "pcr", "plaquetas"]]
          .mean().round(2).rename(index={0: "sem sepse", 1: "com sepse"}))


if __name__ == "__main__":
    main()
