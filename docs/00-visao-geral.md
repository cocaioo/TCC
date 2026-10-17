# 00 — Visão geral do estudo

> Resumo do projeto `Projeto_sepse_07_07.pdf` (seção 3 — Metodologia, pp. 36–47).
> Use estes arquivos em vez de consultar o PDF.

## Índice dos documentos
- [00-visao-geral.md](00-visao-geral.md) — contexto, população, desfecho, fluxo KDD
- [01-dados-variaveis.md](01-dados-variaveis.md) — variáveis preditoras (Quadros 2 e 3)
- [02-algoritmos.md](02-algoritmos.md) — algoritmos (Quadro 4) e protocolo de validação
- [03-metricas-explicabilidade.md](03-metricas-explicabilidade.md) — métricas (Quadro 5), SHAP, desbalanceamento
- [04-pontos-de-atencao.md](04-pontos-de-atencao.md) — inconsistências e riscos identificados no texto

## Tipo de estudo (p. 36)
Coorte retrospectiva com aprendizado de máquina (classificação; o texto também cita regressão).

## Local e população (p. 37)
- UTIN da Nova Maternidade Dona Evangelina Rosa (NMDER), Teresina-PI — 30 leitos.
- Período: 2023–2025 (mínimo de 3 anos).
- Recém-nascidos internados na UTIN por > 72 h.
- Amostra mínima estimada: **~700 pacientes**; incidência esperada de sepse tardia: **15–25%** (≈ 105–175 positivos).

## Critérios (pp. 37–38)
**Inclusão:** internação na UTIN > 72 h; IG ≥ 22 semanas; dados em prontuário eletrônico; sem diagnóstico de sepse precoce nos 3 primeiros dias de vida.

**Exclusão:** malformações congênitas graves ou síndromes genéticas que comprometam a imunidade; óbito nas primeiras 72 h; dados incompletos para preditores essenciais; transferência antes de 72 h.

## Variável-alvo (p. 38)
- **Sepse neonatal tardia**, binária (sim/não) durante a internação.
- Definida pela evolução médica com CID-10 **A41.9** (septicemia não especificada) ou **P36** (septicemia bacteriana do RN).

## Fonte e extração dos dados (pp. 42–44)
- Prontuário **SOUL MV** (MV Sistemas), extraído via API (API Keys) → **CSV**.
- Rotas/campos ainda são *placeholders* (ex.: `/v1/internacoes`) — conector precisa ser adaptado quando sair a especificação oficial.
- Cada RN recebe **ID sequencial** anonimizado na origem (LGPD).
- Escores (nSOFA, NEWS2, qSOFA) **não vêm prontos**: a API dá os componentes brutos e o pipeline deve calculá-los (etapa de transformação).
- Sinais vitais / evolução: última evolução diária documentada (médica e/ou enfermagem).
- Laboratório/microbiologia: módulo de exames, com data/hora de coleta.
- Auditoria manual semanal da base.

## Ambiente (p. 44)
Python ≥ 3.11 (Google Colab), Pandas, NumPy, Scikit-Learn.

## Fluxo KDD (Figura 3, p. 46)
1. **Seleção** — alvo + preditores, extração via API SOUL MV (CSV).
2. **Pré-processamento** — limpeza, imputação, validação manual, derivação de nSOFA/NEWS2/qSOFA.
3. **Transformação** — normalização, construção de atributos, reamostragem para desbalanceamento.
4. **Mineração** — treino dos algoritmos, validação cruzada k-fold (k = 5).
5. **Avaliação** — métrica principal F1-score + AUROC; seleção do melhor modelo.
6. **Interpretação** — SHAP no melhor modelo.

## Teste piloto (pp. 41–42)
Amostra reduzida, aleatória, **não incluída na análise final**, com holdout 70/30. Objetivo: validar o instrumento de coleta, consistência dos registros e tempo de preenchimento.

## Ética (p. 47)
Res. CNS 466/2012 e 510/2016 (dispensa de TCLE), LGPD. CEP da NMDER e CEP/UFPI. Dados anonimizados antes da análise, servidor seguro.

## Cronograma relevante (pp. 47–48)
- Coleta de dados: mai–ago/2027.
- Processamento: set/2027. Análise: out/2027. Defesa: fev/2028.
