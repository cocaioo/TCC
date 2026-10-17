# 01 — Dados e variáveis preditoras

> Fonte: seção 3.5 e Quadros 2 e 3 (pp. 38–40) do `Projeto_sepse_07_07.pdf`.
> A coluna "nome sugerido" é uma proposta para o schema do pipeline — **não está no projeto** e deve ser ajustada aos campos reais do SOUL MV.

## Regra geral (p. 38)
A mineração dos preditores fica restrita aos dados disponíveis **na data da última evolução** do paciente no prontuário. Cinco categorias:
1. Demográficas e do parto
2. Clínicas e de internação
3. Laboratoriais
4. Microbiológicas
5. Escores clínicos

## Janela temporal dos exames laboratoriais (pp. 39–40)
- Referência (T0): **momento da suspeita clínica de sepse tardia** = quando a equipe solicitou a hemocultura.
- Para exames seriados: **última medida antes da suspeita** + **primeiras medidas até 24 h após** T0.
- Literatura citada mostra ganho entre T0 e T6 (6 h após a hemocultura).
- Sem medida na janela → valor **ausente** → imputação no pré-processamento.
- Objetivo explícito: evitar **vazamento temporal** (*data leakage*).
- ⚠️ Para pacientes **sem** sepse não há "suspeita" definida — ver [04-pontos-de-atencao.md](04-pontos-de-atencao.md).

---

## Quadro 2 — Demográficas e do parto (p. 39)

| Variável | Tipo | Nome sugerido |
|---|---|---|
| Sexo do RN (masculino/feminino) | categórica binária | `sexo` |
| Idade gestacional ao nascer (semanas completas) | numérica | `idade_gestacional_sem` |
| Peso ao nascer (g) | numérica | `peso_nascer_g` |
| Apgar no 1º minuto | numérica ordinal (0–10) | `apgar_1min` |
| Apgar no 5º minuto | numérica ordinal (0–10) | `apgar_5min` |
| Tipo de parto (vaginal/cesariana) | categórica binária | `tipo_parto` |
| Idade materna | numérica | `idade_materna` |
| Raça/cor do RN | categórica | `raca_cor_rn` |
| Raça/cor da mãe | categórica | `raca_cor_mae` |
| Número de gestações anteriores | numérica | `n_gestacoes_anteriores` |
| Intercorrências gestacionais: ITU materna | binária | `itu_materna` |
| Intercorrências gestacionais: rotura prematura de membranas | binária | `rotura_prematura_membranas` |
| Intercorrências gestacionais: corioamnionite | binária | `corioamnionite` |
| Intercorrências gestacionais: pré-eclâmpsia | binária | `pre_eclampsia` |
| Uso de antibiótico materno intraparto | binária | `atb_materno_intraparto` |

## Quadro 2 — Clínicas e de internação (p. 39)

| Variável | Tipo | Nome sugerido |
|---|---|---|
| Tempo total de permanência na UTIN (dias) ⚠️ | numérica | `dias_utin` |
| Ventilação mecânica invasiva (sim/não + nº de dias) | binária + numérica | `vmi`, `vmi_dias` |
| Cateter venoso central (sim/não + dias de permanência) | binária + numérica | `cvc`, `cvc_dias` |
| Cateter umbilical (sim/não) | binária | `cateter_umbilical` |
| Nutrição parenteral total (sim/não + nº de dias) | binária + numérica | `npt`, `npt_dias` |
| Antibióticos prévios à sepse tardia (sim/não + classes) | binária + categórica multi-rótulo | `atb_previo`, `atb_previo_classes` |
| Histórias sugestivas de infecção (sim/não) | binária | `historia_infeccao` |
| Critérios de disfunção orgânica (sim/não) | binária | `disfuncao_organica` |
| Enterocolite necrosante (sim/não) | binária | `enterocolite_necrosante` |
| Hemorragia intraventricular (graus I–IV) | ordinal (0–4) | `hiv_grau` |
| Temperatura corporal (mín., máx., média na internação) | numérica ×3 | `temp_min`, `temp_max`, `temp_media` |
| Frequência cardíaca | numérica | `fc` |
| Frequência respiratória | numérica | `fr` |
| SpO2 (oximetria de pulso) | numérica | `spo2` |
| Pressão arterial sistólica | numérica | `pas` |
| Pressão arterial diastólica | numérica | `pad` |
| Vasopressor (sim/não + droga) | binária + categórica | `vasopressor`, `vasopressor_droga` |
| Surfactante exógeno (sim/não) | binária | `surfactante` |
| Episódios de apneia e bradicardia (nº / 24 h) | numérica | `apneia_bradicardia_24h` |
| Intervalo até diagnóstico de sepse (dias desde a internação) ⚠️ | numérica | `dias_ate_diagnostico` |

## Quadro 3 — Laboratoriais (p. 40)

| Variável | Nome sugerido |
|---|---|
| Leucócitos totais | `leucocitos` |
| Neutrófilos segmentados | `neutrofilos_segmentados` |
| Bastões | `bastoes` |
| Relação Imaturos/Totais (I/T, índice de Rodwell) | `relacao_it` |
| Plaquetas | `plaquetas` |
| Proteína C Reativa (PCR), quantitativa | `pcr` |
| Procalcitonina (PCT) — quando disponível | `pct` |
| Glicemia | `glicemia` |
| Creatinina sérica | `creatinina` |
| Bilirrubina total e frações | `bilirrubina_total`, `bilirrubina_direta`, `bilirrubina_indireta` |
| pH sanguíneo (arterial ou venoso) | `ph` |
| Lactato sérico | `lactato` |
| Hemoglobina | `hemoglobina` |
| Hematócrito | `hematocrito` |
| Sódio, potássio, cálcio ionizado | `sodio`, `potassio`, `calcio_ionizado` |
| TGO/TGP (aminotransferases) | `tgo`, `tgp` |
| Relação Neutrófilos/Linfócitos | `relacao_nl` |

Cada exame seriado pode virar 2 colunas no pipeline: `<exame>_pre` (última antes de T0) e `<exame>_pos24h` (primeira até 24 h após).

## Quadro 3 — Microbiológicas (p. 40)

| Variável | Nome sugerido |
|---|---|
| Resultado da hemocultura (positiva/negativa + agente isolado) ⚠️ | `hemocultura_positiva`, `hemocultura_agente` |
| Perfil de sensibilidade antimicrobiana ⚠️ | `perfil_sensibilidade` |
| Histórico de infecções relacionadas à assistência à saúde | `historico_iras` |
| Colonização por *Staphylococcus* coagulase-negativo ou *Klebsiella* sp. | `colonizacao_scn`, `colonizacao_klebsiella` |

## Quadro 3 — Escores clínicos (p. 40)

| Escore | Nome sugerido | Observação |
|---|---|---|
| Escala de Coma de Glasgow neonatal adaptada | `glasgow_neonatal` | — |
| NEWS2 (National Early Warning Score) | `news2` | calculado no pipeline a partir dos componentes |
| qSOFA (Quick SOFA) | `qsofa` | calculado no pipeline a partir dos componentes |
| nSOFA (citado nas pp. 43 e 46, **ausente do Quadro 3**) | `nsofa` | calculado no pipeline |

Os escores **não vêm prontos** do SOUL MV (p. 43): a derivação faz parte da etapa de transformação e deve ser validada explicitamente.

⚠️ = ver [04-pontos-de-atencao.md](04-pontos-de-atencao.md).
