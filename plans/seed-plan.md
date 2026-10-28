# Plano: gerador de dados sintéticos (seed) para o pipeline de sepse neonatal tardia

| | |
|---|---|
| **Status** | Implementado e testado (17 testes passando) |
| **Branch** | `create-pipeline` |
| **Criado / atualizado** | 2026-10-28 |
| **Código** | `gerador_seed.py`, `tests/test_gerador_seed.py` |
| **Fonte dos requisitos** | `Projeto_sepse_07_07.pdf`, Quadros 2 e 3 (págs. 39–40), mais págs. 33–34, 37–38, 40, 43–46 |

Este documento é versionado junto com o código para dar rastreabilidade: o que foi decidido, por quê, o que foi
implementado e o que ficou pendente. Toda mudança relevante entra no [histórico](#10-histórico-de-mudanças).

---

## 1. Contexto e objetivo

O TCC vai treinar os algoritmos do Quadro 4 (GB, XGBoost, DT, KNN, RF, NB, RL, MLP) para prever **sepse neonatal
tardia** (CID-10 A41.9 / P36) com dados da UTIN, validação cruzada k = 5, métrica principal F1-score e explicabilidade
por SHAP. Os dados reais dependem de aprovação em CEP e do conector com o prontuário SOUL MV, ainda não disponíveis.

O objetivo do seed é ter um CSV **fictício, com formato parecido com o real**, para construir e testar o pipeline KDD
(pré-processamento → transformação → mineração → avaliação → SHAP) antes dos dados reais.

**Não é objetivo:** produzir métricas com valor científico. Os modelos vão "descobrir" o que foi plantado no gerador.

## 2. Decisões tomadas

| # | Decisão | Escolha | Motivo | Quem decidiu |
|---|---|---|---|---|
| D1 | Formato da tabela | 1 linha por RN, tabela plana (CSV único) | É o que os algoritmos do Quadro 4 consomem direto; o PDF (p. 43) prevê exportação tabular em CSV | Usuário |
| D2 | Realismo | Sinal clínico plausível (variáveis ligadas ao desfecho e entre si), com ruído e ausentes | Aleatório independente só testaria o "encanamento"; métricas ficariam no acaso | Usuário |
| D3 | Origem dos pesos | Palpites clínicos do autor do código, em tabelas no topo do módulo, marcados "a revisar" | Rápido e suficiente para testar o pipeline; ORs da literatura ficam como evolução possível | Usuário |
| D4 | Escopo de colunas | **Manter as 99 colunas** e colapsar depois, se necessário | Colapsar na etapa de transformação é fácil; expandir exigiria mexer no gerador | Usuário |
| D5 | Forma do código | Módulo `.py` + CLI, sem notebook | Preferência registrada do usuário (memória: preferências de trabalho) | Usuário |
| D6 | Prevalência | Exata (`round(n·prevalencia)` maiores escores viram sépticos), padrão 20% | PDF prevê 15–25%; evita bisseção e testes instáveis | Revisão adversarial |
| D7 | Intensidade do sinal | `--sinal` padrão **1.0** (plano inicial dizia 0.6) | Com 1.0 o PCR sozinho dá AUC ≈ 0,8, razoável para um marcador de sepse; 0.6 fica como cenário difícil | Implementação |
| D8 | Nomes dos exames seriados | Sufixos `_pre` e `_pos24h` (não `_t6`) | Na p. 40, T0/T6 refere-se aos escores; para exames a janela é "última medida antes" e "primeiras até 24 h depois" | Revisão adversarial |
| D9 | nSOFA | **Fora desta versão** | Não consta no Quadro 3; faltam pontos de corte e FiO2; uma versão "aproximada" seria um escore inventado | Revisão adversarial |
| D10 | qSOFA e NEWS2 | Derivados dos componentes, com cortes de adulto, como o PDF manda (p. 43) | Fidelidade ao PDF; a limitação é documentada (ver seção 8) | PDF |
| D11 | Ausentes | Só em exames e hemocultura, MCAR; demográficas e clínicas completas | Exercita a imputação do pré-processamento; o PDF exclui prontuários sem variáveis essenciais | Implementação |
| D12 | `dias_ate_dx_sepse` | NaN nos não sépticos | Tal como definido no PDF (intervalo até o diagnóstico); gera vazamento de alvo, sinalizado | Implementação |
| D13 | Convenções de formato | Flags 0/1; categóricas em texto sem acento; colunas em snake_case sem acento | Encoding fica para a etapa de transformação do KDD | Revisão adversarial |
| D14 | Dependências | Só `numpy` e `pandas` (já em `requirements.txt`) | Sem scikit-learn: AUC dos testes calculada por postos | Implementação |

## 3. O que foi implementado

### 3.1 Arquivos

| Arquivo | Conteúdo |
|---|---|
| `gerador_seed.py` | Módulo + CLI. API pública: `gerar_seed(n, prevalencia, sinal, ausentes, seed) -> DataFrame` |
| `tests/test_gerador_seed.py` | 17 testes pytest |
| `data/seed_sepse.csv` | Saída (UTF-8). Ignorada pelo git (`data/` está no `.gitignore`); regenerável |

### 3.2 CLI

```
.venv\Scripts\python.exe gerador_seed.py --n 1000 --prevalencia 0.20 --sinal 1.0 --ausentes 1.0 --seed 42 --saida data/seed_sepse.csv
```

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `--n` | 1000 | Nº de recém-nascidos (PDF: mínimo de 700) |
| `--prevalencia` | 0.20 | Proporção de sepse tardia (PDF: 15–25%); valida 0 < p < 1 |
| `--sinal` | 1.0 | 0 = classes indistinguíveis; 1 = efeitos de `EFEITO_DP` por inteiro; valida 0 ≤ s ≤ 1 |
| `--ausentes` | 1.0 | Multiplicador das taxas de `TAXA_AUSENTE` (0 = sem ausentes) |
| `--seed` | 42 | Semente de `np.random.default_rng`; mesma semente = CSV idêntico |
| `--saida` | `data/seed_sepse.csv` | Caminho; a pasta é criada se não existir |

### 3.3 Como a aleatoriedade é construída (ordem de geração)

1. **Exposições de parto e internação**, sempre com dependências, nunca colunas isoladas:
   - IG ~ N(32,5; 4,2) arredondada, entre 22 e 41; peso = média(IG) × (1 + 0,12·N(0,1)) com média interpolada por IG;
   - Apgar menor nos mais prematuros, com `apgar_5min ≥ apgar_1min`;
   - `atb_materno_intraparto` mais provável com corioamnionite/RPMO/ITU; RPMO e pré-eclâmpsia mais prováveis se IG < 34;
   - `dias_utin ≥ 4` (inclusão > 72 h), crescente com a prematuridade;
   - VM, CVC, NPT, cateter umbilical, surfactante e ECN mais prováveis com menor IG;
   - `*_dias ≤ dias_utin`, e `0` quando a flag é 0; HIV só em IG < 32; classes de antibiótico só se `atb_previo = 1`, com
     ao menos uma ativa.
2. **Desfecho**: `escore = Σ PESOS_RISCO · fatores + ruído logístico (rng.logistic)`; os `round(n·prevalencia)` maiores
   escores recebem `sepse_tardia = 1`. Modelo logístico, prevalência exata, sem calibração por bisseção.
3. **Manifestações condicionadas ao desfecho**: cada variável contínua tem um efeito `d` em desvios-padrão
   (`EFEITO_DP`) aplicado como `d · sinal` nos sépticos. AUC univariada esperada = Φ(|d|·sinal / √2). Variáveis
   positivas e assimétricas (PCR, lactato, leucócitos, plaquetas etc.) em escala log-normal; as simétricas
   (temperatura, Na, pH) em normal; tudo recortado em faixa fisiológica. Flags (vasopressor, disfunção orgânica, história
   sugestiva, hemocultura positiva) usam probabilidade que vai de `p_base` a `p_sepse` conforme `sinal`.
   A segunda medida do exame (`_pos24h`) = primeira + 30% do deslocamento + ruído serial (40% do dp).
4. **Coerência fisiológica**: `temp_min ≤ temp_media ≤ temp_max`; `pad < pas`; `bilirrubina_total = direta + indireta`;
   `hematocrito ≈ 3·hemoglobina`; neutrófilos (seg + bastões) ≤ leucócitos; `dias_ate_dx_sepse` entre 3 e `dias_utin`.
5. **Ausentes**: máscara MCAR por coluna (`TAXA_AUSENTE`; `_pos24h` soma 10 p.p.; teto de 95%); hemocultura ausente
   (3%) leva junto agente e perfil de sensibilidade.
6. **Derivadas calculadas depois dos ausentes**, a partir das colunas já com NaN (o NaN se propaga como no conector
   real): `relacao_it`, `relacao_neutrofilos_linfocitos`, `bilirrubina_indireta` (= total − direta). `qsofa` e `news2`
   vêm de sinais vitais e Glasgow, que não têm ausentes, então ficam completos.

### 3.4 Colunas geradas (99 = 97 preditoras + `id_paciente` + `sepse_tardia`)

| Grupo | Colunas | Nº |
|---|---|---|
| Identificação / alvo | `id_paciente` (RN0001…), `sepse_tardia` (0/1, última coluna) | 2 |
| Demográficas e do parto | `sexo`, `idade_gestacional_sem`, `peso_nascer_g`, `apgar_1min`, `apgar_5min`, `tipo_parto`, `idade_materna`, `raca_cor_rn`, `raca_cor_mae`, `n_gestacoes_anteriores`, `itu_materna`, `rpmo`, `corioamnionite`, `pre_eclampsia`, `atb_materno_intraparto` | 15 |
| Clínicas e de internação | `dias_utin`, `vm_invasiva`, `vm_dias`, `cvc`, `cvc_dias`, `cateter_umbilical`, `npt`, `npt_dias`, `atb_previo`, `atb_prev_penicilinas`, `_cefalosporinas`, `_aminoglicosideos`, `_carbapenemicos`, `_glicopeptideos`, `ecn`, `hiv_grau` (0–4), `surfactante` | 17 |
| Evolução e sinais vitais | `historia_sugestiva_infeccao`, `criterios_disfuncao_organica`, `temp_min`, `temp_max`, `temp_media`, `fc`, `fr`, `spo2`, `pas`, `pad`, `vasopressor`, `apneia_bradicardia_24h`, `glasgow_neonatal` | 13 |
| Intervalo até o diagnóstico | `dias_ate_dx_sepse` (NaN nos não sépticos) | 1 |
| Laboratoriais (22 × 2) | `leucocitos`, `neutrofilos_seg`, `bastoes`, `relacao_it`, `plaquetas`, `pcr`, `pct`, `glicemia`, `creatinina`, `bilirrubina_total`, `_direta`, `_indireta`, `ph`, `lactato`, `hemoglobina`, `hematocrito`, `sodio`, `potassio`, `calcio_ionico`, `tgo`, `tgp`, `relacao_neutrofilos_linfocitos`, cada uma com `_pre` e `_pos24h` | 44 |
| Microbiológicas | `hemocultura_resultado`, `hemocultura_agente`, `perfil_sensibilidade` (só em positivas), `hist_infeccao_ira`, `colonizacao` (nenhuma / scn / klebsiella) | 5 |
| Escores | `qsofa`, `news2` (o Glasgow já foi contado acima) | 2 |

Unidades: leucócitos, neutrófilos, bastões e plaquetas em /mm³; PCR mg/L; PCT ng/mL; glicemia, creatinina e
bilirrubinas mg/dL; lactato mmol/L; Hb g/dL; Ht %; Na e K mEq/L; Ca ionizado mmol/L; TGO/TGP U/L; temperatura °C.

### 3.5 Tabelas de parâmetros (todas palpites, **a revisar**)

Ficam no topo de `gerador_seed.py`: `PESOS_RISCO`, `PARAMETROS`, `EFEITO_DP`, `PROB_FLAG_SEPSE`, `TAXA_AUSENTE`,
`CASAS_DECIMAIS`, `CLASSES_ATB`, `VARS_RISCO_VAZAMENTO`.

Pesos do logit do desfecho (log do odds ratio; 0,4 ≈ OR 1,5; 0,7 ≈ OR 2,0; 0,9 ≈ OR 2,5):

| Fator | Peso |
|---|---|
| IG < 28 semanas | 0,9 |
| CVC | 0,7 |
| ECN | 0,6 |
| NPT | 0,5 |
| VM invasiva | 0,4 |
| Antibiótico prévio | 0,4 |
| Cateter umbilical | 0,3 |
| A cada 10 dias de UTIN | 0,3 |

Efeito nos sépticos em desvios-padrão (`EFEITO_DP`, com `sinal = 1`): PCT +1,5; PCR +1,4; lactato +1,0; FC +0,9;
plaquetas −0,9; temperatura +0,8; pH −0,8; SpO2 −0,8; PAS −0,8; frações de bastões +0,8; FR +0,7; leucócitos +0,6;
creatinina +0,5; glicemia, linfócitos, sódio, cálcio ionizado, TGO e bilirrubina direta ±0,4; hemoglobina e TGP ±0,3;
potássio +0,2; bilirrubina indireta +0,1; Glasgow −1,2. Os demais valores (probabilidades das flags, taxas de ausência)
estão no código.

## 4. Variáveis do PDF × inferências

O PDF lista **52 itens** (Quadro 2: 10 demográficas e do parto + 20 clínicas e de internação; Quadro 3: 15
laboratoriais + 4 microbiológicas + 3 escores). Chegamos a 99 colunas por desdobramento de itens compostos e por
duplicação dos exames. **Nenhuma variável fora do PDF foi acrescentada; só o formato foi inferido.**

### 4.1 Desdobramentos que o PDF pede explicitamente

- Apgar no 1º e 5º minuto → `apgar_1min`, `apgar_5min`.
- Raça ou cor do recém-nascido e da mãe → `raca_cor_rn`, `raca_cor_mae`.
- Intercorrências gestacionais (4 nomeadas no quadro) → `itu_materna`, `rpmo`, `corioamnionite`, `pre_eclampsia`.
- VM, CVC e NPT com "sim/não e número de dias" → flag + `_dias`.
- Temperatura "mínimo, máximo e médio" → `temp_min`, `temp_max`, `temp_media`.
- Neutrófilos segmentados e bastões; bilirrubina total e frações; sódio, potássio e cálcio ionizado; hemoglobina e
  hematócrito; TGO/TGP → uma coluna por componente.

### 4.2 Inferências do autor do código (o PDF não define)

Para registrar quais colunas dependem de interpretação, esta lista deve virar a constante `INFERIDO_DO_PDF` no código
(ver pendências, seção 9).

| # | Inferência | Colunas | Observação |
|---|---|---|---|
| I1 | Duas medidas por exame | os 22 exames × (`_pre`, `_pos24h`) | Maior inferência (+22 colunas). Baseada na p. 40; o PDF não diz que são duas variáveis. Alternativas ao colapsar: só a medida mais próxima da suspeita, ou a variação entre as duas |
| I2 | 5 classes de antibiótico | `atb_prev_penicilinas`, `_cefalosporinas`, `_aminoglicosideos`, `_carbapenemicos`, `_glicopeptideos` | PDF diz "classes utilizadas" sem listar |
| I3 | Categorias de agente e sensibilidade | `hemocultura_agente` (scn, staphylococcus_aureus, klebsiella, e_coli, candida, enterococcus, pseudomonas, outros), `perfil_sensibilidade` (sensivel, resistente_oxacilina, multirresistente) | PDF cita "agente isolado" e "perfil de sensibilidade" sem categorias |
| I4 | `colonizacao` em uma coluna de 3 valores | `colonizacao` | O PDF poderia pedir duas flags separadas (SCN, Klebsiella) |
| I5 | Categorias de raça/cor | `raca_cor_rn`, `raca_cor_mae` | Padrão IBGE (parda, branca, preta, amarela, indigena); não listadas no PDF |
| I6 | Índice IT só com bastões | `relacao_it_*` | O índice de Rodwell real inclui outros imaturos (metamielócitos, mielócitos), ausentes do quadro |
| I7 | Proxy de O2 suplementar no NEWS2 | `news2` | Sem coluna de O2 nas variáveis do projeto; `vm_invasiva` faz o papel (2 pontos) |

Consequência: I2 a I5 são nomes e categorias fictícias. Com os dados reais do SOUL MV o conjunto será outro e o
conector precisará ser adaptado. Não tratar essas colunas como "o que o PDF define".

## 5. Resultado da execução de referência

`gerar_seed(n=1000, seed=42)` com os parâmetros padrão: 1000 linhas × 99 colunas, 20,0% de sepse tardia.

| Variável | Sem sepse | Com sepse |
|---|---|---|
| IG média (semanas) | 33,0 | 29,8 |
| `dias_utin` médio | 19,2 | 30,1 |
| CVC | 26% | 52% |
| `pcr_pre` médio (mg/L) | 4,3 | 15,3 |
| `lactato_pre` médio (mmol/L) | 2,0 | 3,3 |
| `plaquetas_pre` médias (/mm³) | 240.095 | 184.477 |
| PAS média (mmHg) | 64,6 | 54,0 |
| Vasopressor | 2% | 28% |
| `qsofa` / `news2` médios | 2,4 / 11,7 | 2,9 / 14,3 |

Hemocultura: 10% positivas (100 de 1000), 2% ausentes. Ausentes em exames: ~2% a 15% na medida `_pre`, até ~57% (PCT)
na `_pos24h`. Distribuições gerais plausíveis (peso médio ≈ 2.000 g; 41% com IG < 32; apgar 1º min médio 7,4).

## 6. Verificação

`.venv\Scripts\python.exe -m pytest` → **17 passed** (cerca de 2 s). Cobertura:

- colunas dos Quadros 2 e 3 presentes, sem duplicatas;
- mesma semente → DataFrame idêntico; semente diferente → diferente;
- prevalência exata e validação de argumentos (`prevalencia`, `sinal`);
- critérios de inclusão: `dias_utin > 3`, 22 ≤ IG ≤ 41;
- coerência: `*_dias ≤ dias_utin` e 0 com flag 0; `apgar_5min ≥ apgar_1min`; HIV só se IG < 32; classes de ATB só com
  `atb_previo = 1`; vasopressor implica disfunção orgânica;
- faixas fisiológicas (`temp_min ≤ media ≤ max`, `pad < pas`, SpO2, Glasgow, peso, pH);
- relações entre exames (bilirrubinas, neutrófilos ≤ leucócitos, IT entre 0 e 1);
- escores nos intervalos (`qsofa` 0–3, `news2` 0–20);
- `dias_ate_dx_sepse` só nos sépticos e entre 3 e `dias_utin`;
- AUC univariada (por postos, sem sklearn) de PCR, lactato, plaquetas e PCT ≈ Φ(d·sinal/√2) com tolerância de 0,05;
  com `sinal = 0`, AUC ≈ 0,5 (inclui FC);
- fração de ausentes controlada por `--ausentes` (0 = nenhum; PCT ≈ 45%; leucócitos ≈ 2%; `_pos24h` > `_pre`);
- microbiologia coerente (agente só se positiva; positivas entre 15% e 45% dos sépticos).

## 7. Riscos metodológicos do PDF (não do seed)

- **Vazamento por tempo de internação.** `dias_utin`, `vm_dias`, `cvc_dias` e `npt_dias` são totais da internação toda, e
  a sepse prolonga a internação. No dado real podem ser em parte consequência do desfecho; no seed só o causam.
- **Vazamento por variáveis do próprio diagnóstico.** `dias_ate_dx_sepse` só existe para quem teve sepse; `hemocultura_*`
  está diretamente ligada ao diagnóstico. Todas estão em `VARS_RISCO_VAZAMENTO` para decisão no pré-processamento.
- **Referência temporal contraditória.** A p. 38 restringe os preditores aos "dados da última evolução"; a p. 40 define
  o momento da suspeita (hemocultura) para evitar vazamento. Para quem não teve sepse, o T0 não está definido; o seed
  gera tudo como se houvesse um momento de suspeita para todos.
- **Dados incompletos.** A p. 38 exclui prontuários com variáveis incompletas; a p. 40 prevê imputação. O seed mantém NaN
  só em exames e hemocultura, para exercitar a imputação.
- **Poucos casos por variável.** ~700 pacientes × 15–25% de sepse = 105 a 175 positivos, contra 97 preditores: 1 a 2
  positivos por variável. Risco de sobreajuste, sobretudo em árvores, KNN e MLP. O PDF prevê validação cruzada e SHAP,
  mas não seleção de variáveis. Levar à orientadora. O seed permite testar o efeito (ex.: com e sem as colunas `_pos24h`).

## 8. Limitações do seed

- Pesos e efeitos são palpites, não estimados de dados reais (ORs da literatura poderiam ancorá-los; fontes candidatas
  citadas no PDF: Brito et al. 2024, Huncikova et al. 2023, Coggins & Glaser 2022; **não foram lidas**).
- Os modelos vão "descobrir" exatamente o que foi plantado; o SHAP vai destacar CVC, PCR e prematuridade porque foram
  programados assim. Serve para validar o pipeline de ponta a ponta, não para tirar conclusões clínicas.
- As relações reais são mais complexas (interações, ausência informativa); o desempenho com dados reais deve ser pior.
- Ausentes são MCAR; na prática a ausência costuma depender da gravidade (ex.: PCT pedida a quem parece mais grave).
- **qSOFA e NEWS2 com cortes de adulto quase não variam em neonatos.** No seed 42, `qsofa` fica entre 1 e 3 e `news2`
  entre 6 e 17: FR ≥ 22 e PAS ≤ 100 valem para praticamente todo neonato, e FC/FR neonatais normais já pontuam alto.
  É coerente com a ressalva do próprio PDF (p. 33) e foi implementado como o PDF manda. Na prática esses escores
  carregam pouca informação neste conjunto.

## 9. Pendências e próximos passos

- [ ] Registrar `INFERIDO_DO_PDF` em `gerador_seed.py` com as colunas das inferências I1 a I7 (seção 4.2).
- [ ] nSOFA: está na Fig. 3 e na p. 43 como escore a derivar, mas não no Quadro 3. Entra depois de conferir a fórmula em
      Wynn & Polin (2020) e de haver coluna de FiO2.
- [ ] Opcional: ancorar `PESOS_RISCO` e `EFEITO_DP` em ORs da literatura (opção 2 discutida), ou pedir ORs à orientadora.
- [ ] Opcional: ausência informativa (MAR) e "sepse silenciosa" (sépticos com exames normais) para dificultar o problema.
- [ ] Levar à orientadora: riscos da seção 7 (vazamento, T0 indefinido, poucos casos por variável).
- [ ] Próxima etapa do pipeline (fora deste plano): pré-processamento/imputação, depois transformação, modelos e SHAP,
      uma etapa por vez, com explicação dos conceitos antes do código.

Observação: o PDF descreve os algoritmos do Quadro 4 como ML supervisionado clássico (não IA generativa); o CSV tabular
atende a eles.

## 10. Histórico de mudanças

| Data | Mudança |
|---|---|
| 2026-10-28 | Plano inicial aprovado (formato plano, sinal clínico, pesos como palpites) |
| 2026-10-28 | Revisão adversarial: `_t6` → `_pos24h`; prevalência exata por ranking; nSOFA retirado; `EFEITO_DP` em desvios-padrão; unidades; lista de vazamentos |
| 2026-10-28 | Implementação de `gerador_seed.py` e `tests/test_gerador_seed.py`; 17 testes passando |
| 2026-10-28 | `--sinal` padrão 1.0 (era 0.6); `bilirrubina_indireta` derivada; `qsofa`/`news2` ficam sem ausentes |
| 2026-10-28 | Decisão D4: manter as 99 colunas e colapsar só se necessário |
| 2026-10-28 | Registro das variáveis do PDF × inferências (seção 4) e reescrita completa deste documento |
