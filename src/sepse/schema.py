"""Dicionário das variáveis do estudo (Quadros 2 e 3 do projeto).

Fonte única da verdade para nomes, tipos e faixas válidas das colunas.
Ver docs/01-dados-variaveis.md e docs/04-pontos-de-atencao.md.
"""

from dataclasses import dataclass

ID = "id_paciente"
ALVO = "sepse_tardia"

# Tipos
BINARIA = "binaria"  # 0/1
ORDINAL = "ordinal"  # inteiro com ordem (Apgar, grau de HIV, escores)
NUMERICA = "numerica"  # contínua
CATEGORICA = "categorica"  # texto com valores fixos

# Categorias (grupos do projeto)
DEMOGRAFICA = "demografica"
CLINICA = "clinica"
LABORATORIAL = "laboratorial"
MICROBIOLOGICA = "microbiologica"
ESCORE = "escore"

RACAS = ("branca", "preta", "parda", "amarela", "indigena")
CLASSES_ATB = ("penicilina+aminoglicosideo", "cefalosporina", "glicopeptideo", "carbapenem")
VASOPRESSORES = ("dopamina", "dobutamina", "adrenalina", "noradrenalina")
AGENTES = ("scn", "klebsiella", "s_aureus", "e_coli", "candida")
PERFIS_SENSIBILIDADE = ("sensivel", "multirresistente")


@dataclass(frozen=True)
class Coluna:
    nome: str
    tipo: str
    categoria: str
    minimo: float | None = None
    maximo: float | None = None
    valores: tuple[str, ...] | None = None  # só para categóricas
    suspeita_vazamento: bool = False  # ver docs/04-pontos-de-atencao.md


def _bin(nome, categoria, **kw):
    return Coluna(nome, BINARIA, categoria, 0, 1, **kw)


COLUNAS: tuple[Coluna, ...] = (
    # --- Quadro 2: demográficas e do parto ---
    Coluna("sexo", CATEGORICA, DEMOGRAFICA, valores=("M", "F")),
    Coluna("idade_gestacional_sem", ORDINAL, DEMOGRAFICA, 22, 42),
    Coluna("peso_nascer_g", NUMERICA, DEMOGRAFICA, 300, 5500),
    Coluna("apgar_1min", ORDINAL, DEMOGRAFICA, 0, 10),
    Coluna("apgar_5min", ORDINAL, DEMOGRAFICA, 0, 10),
    Coluna("tipo_parto", CATEGORICA, DEMOGRAFICA, valores=("vaginal", "cesariana")),
    Coluna("idade_materna", ORDINAL, DEMOGRAFICA, 12, 50),
    Coluna("raca_cor_rn", CATEGORICA, DEMOGRAFICA, valores=RACAS),
    Coluna("raca_cor_mae", CATEGORICA, DEMOGRAFICA, valores=RACAS),
    Coluna("n_gestacoes_anteriores", ORDINAL, DEMOGRAFICA, 0, 15),
    _bin("itu_materna", DEMOGRAFICA),
    _bin("rotura_prematura_membranas", DEMOGRAFICA),
    _bin("corioamnionite", DEMOGRAFICA),
    _bin("pre_eclampsia", DEMOGRAFICA),
    _bin("atb_materno_intraparto", DEMOGRAFICA),
    # --- Quadro 2: clínicas e de internação ---
    Coluna("dias_utin", ORDINAL, CLINICA, 3, 365, suspeita_vazamento=True),
    _bin("vmi", CLINICA),
    Coluna("vmi_dias", ORDINAL, CLINICA, 0, 365),
    _bin("cvc", CLINICA),
    Coluna("cvc_dias", ORDINAL, CLINICA, 0, 365),
    _bin("cateter_umbilical", CLINICA),
    _bin("npt", CLINICA),
    Coluna("npt_dias", ORDINAL, CLINICA, 0, 365),
    _bin("atb_previo", CLINICA),
    Coluna("atb_previo_classes", CATEGORICA, CLINICA, valores=CLASSES_ATB),
    _bin("historia_infeccao", CLINICA),
    _bin("disfuncao_organica", CLINICA),
    _bin("enterocolite_necrosante", CLINICA),
    Coluna("hiv_grau", ORDINAL, CLINICA, 0, 4),
    Coluna("temp_min", NUMERICA, CLINICA, 33.0, 41.0),
    Coluna("temp_max", NUMERICA, CLINICA, 33.0, 41.0),
    Coluna("temp_media", NUMERICA, CLINICA, 33.0, 41.0),
    Coluna("fc", NUMERICA, CLINICA, 60, 250),
    Coluna("fr", NUMERICA, CLINICA, 10, 120),
    Coluna("spo2", NUMERICA, CLINICA, 50, 100),
    Coluna("pas", NUMERICA, CLINICA, 20, 120),
    Coluna("pad", NUMERICA, CLINICA, 10, 90),
    _bin("vasopressor", CLINICA),
    Coluna("vasopressor_droga", CATEGORICA, CLINICA, valores=VASOPRESSORES),
    _bin("surfactante", CLINICA),
    Coluna("apneia_bradicardia_24h", ORDINAL, CLINICA, 0, 50),
    Coluna("dias_ate_diagnostico", ORDINAL, CLINICA, 3, 365, suspeita_vazamento=True),
    # --- Quadro 3: laboratoriais ---
    Coluna("leucocitos", NUMERICA, LABORATORIAL, 500, 80000),  # /mm³
    Coluna("neutrofilos_segmentados", NUMERICA, LABORATORIAL, 0, 60000),  # /mm³
    Coluna("bastoes", NUMERICA, LABORATORIAL, 0, 20000),  # /mm³
    Coluna("relacao_it", NUMERICA, LABORATORIAL, 0, 1),
    Coluna("plaquetas", NUMERICA, LABORATORIAL, 5000, 900000),  # /mm³
    Coluna("pcr", NUMERICA, LABORATORIAL, 0, 400),  # mg/L
    Coluna("pct", NUMERICA, LABORATORIAL, 0, 200),  # ng/mL
    Coluna("glicemia", NUMERICA, LABORATORIAL, 10, 500),  # mg/dL
    Coluna("creatinina", NUMERICA, LABORATORIAL, 0.1, 5),  # mg/dL
    Coluna("bilirrubina_total", NUMERICA, LABORATORIAL, 0.1, 40),  # mg/dL
    Coluna("bilirrubina_direta", NUMERICA, LABORATORIAL, 0, 20),
    Coluna("bilirrubina_indireta", NUMERICA, LABORATORIAL, 0, 40),
    Coluna("ph", NUMERICA, LABORATORIAL, 6.8, 7.7),
    Coluna("lactato", NUMERICA, LABORATORIAL, 0.3, 20),  # mmol/L
    Coluna("hemoglobina", NUMERICA, LABORATORIAL, 5, 25),  # g/dL
    Coluna("hematocrito", NUMERICA, LABORATORIAL, 15, 75),  # %
    Coluna("sodio", NUMERICA, LABORATORIAL, 110, 170),  # mEq/L
    Coluna("potassio", NUMERICA, LABORATORIAL, 2, 9),  # mEq/L
    Coluna("calcio_ionizado", NUMERICA, LABORATORIAL, 0.6, 1.8),  # mmol/L
    Coluna("tgo", NUMERICA, LABORATORIAL, 5, 2000),  # U/L
    Coluna("tgp", NUMERICA, LABORATORIAL, 5, 2000),  # U/L
    Coluna("relacao_nl", NUMERICA, LABORATORIAL, 0, 40),
    # --- Quadro 3: microbiológicas ---
    _bin("hemocultura_positiva", MICROBIOLOGICA, suspeita_vazamento=True),
    Coluna("hemocultura_agente", CATEGORICA, MICROBIOLOGICA, valores=AGENTES, suspeita_vazamento=True),
    Coluna(
        "perfil_sensibilidade",
        CATEGORICA,
        MICROBIOLOGICA,
        valores=PERFIS_SENSIBILIDADE,
        suspeita_vazamento=True,
    ),
    _bin("historico_iras", MICROBIOLOGICA),
    _bin("colonizacao_scn", MICROBIOLOGICA),
    _bin("colonizacao_klebsiella", MICROBIOLOGICA),
    # --- Quadro 3: escores (nSOFA citado nas pp. 43/46) ---
    Coluna("glasgow_neonatal", ORDINAL, ESCORE, 3, 15),
    Coluna("news2", ORDINAL, ESCORE, 0, 20),
    Coluna("qsofa", ORDINAL, ESCORE, 0, 3),
    Coluna("nsofa", ORDINAL, ESCORE, 0, 15),
)

POR_NOME: dict[str, Coluna] = {c.nome: c for c in COLUNAS}
PREDITORES: tuple[str, ...] = tuple(c.nome for c in COLUNAS)
VAZAMENTO: tuple[str, ...] = tuple(c.nome for c in COLUNAS if c.suspeita_vazamento)
ORDEM_COLUNAS: tuple[str, ...] = (ID, *PREDITORES, ALVO)
