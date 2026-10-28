# Plano: pré-processamento e transformação (etapas 2 e 3 do KDD)

Documento versionado junto com o código (mesmo padrão de `seed-plan.md`); atualizado a cada passo concluído.

| | |
|---|---|
| **Status** | Plano aprovado em 2026-10-28; passos 1 e 2 de 5 concluídos |
| **Branch** | `create-pipeline` |
| **Entrada** | `data/seed_sepse.csv` (99 colunas: 97 preditoras + `id_paciente` + `sepse_tardia`), de `gerador_seed.py` |
| **Saída** | Divisão treino/teste e um pré-processador sklearn **não ajustado**, pronto para os modelos |
| **Fonte no PDF** | Fig. 3 (p. 46: limpeza, imputação, validação, derivação dos escores; normalização, construção de atributos); pp. 40, 44–46 |
| **Código previsto** | `preprocessamento.py`, `tests/test_preprocessamento.py` |

## 1. Contexto e objetivo

O seed gera dados com formato parecido com o real. Antes de treinar os algoritmos do Quadro 4, o PDF prevê limpeza,
imputação dos ausentes (p. 40: "tratado como ausente e submetido aos procedimentos de imputação descritos na etapa de
pré-processamento"), normalização e construção de atributos. Esta etapa entrega:

1. **Auditoria** dos dados (tipos, ausentes, constantes, faixas, critérios de inclusão): o equivalente automatizado da
   "auditoria e validação manual" da p. 43–44.
2. **Divisão treino/teste** estratificada, congelada em disco.
3. **Pré-processador** (`ColumnTransformer` do scikit-learn) que imputa, transforma e codifica, **ajustado só no treino**.

**Fora do escopo** (próximas etapas, uma por vez): reamostragem (SMOTE/pesos de classe), modelos, GridSearchCV,
validação cruzada, métricas e SHAP. A reamostragem **não** entra aqui, porque só pode ser aplicada dentro de cada dobra
da validação cruzada (ver conceito C1).

## 2. Conceitos (para ler antes de aprovar)

**C1. Vazamento de dados no pré-processamento.** Qualquer valor aprendido dos dados (mediana usada para imputar, média e
desvio da padronização, categorias do one-hot, SMOTE) deve vir **só do treino**. Se a mediana for calculada com o conjunto
todo, o teste "vaza" informação para o treino e a métrica final fica otimista. Por isso o pré-processador é entregue
**sem ajuste**: na validação cruzada ele é reajustado dentro de cada dobra (`Pipeline`), nunca antes.

**C2. Divisão antes de tudo, estratificada.** 80% treino / 20% teste (p. 45), preservando a proporção de sepse em cada
parte. Com 1000 pacientes e 20% de prevalência: 800 treino (160 positivos) e 200 teste (40 positivos). O teste só é
aberto uma vez, no fim. (A p. 41 cita 70/30 para o *teste piloto*; para a validação do modelo vale a p. 45, 80/20.)

**C3. Tipos de ausência.** MCAR (aleatória), MAR (depende de outra variável observada) e MNAR (depende do próprio valor).
No hospital real a ausência de PCT é informativa (pede-se PCT a quem parece mais grave). Por isso, além de preencher,
guardamos um **indicador de ausência** (0/1 "foi medido?") para exames com muitos ausentes. No seed a ausência é MCAR, então
os indicadores aqui são ruído; no dado real podem carregar sinal. Há um custo: indicadores demais inflam as colunas
(com ~160 positivos, mais colunas = mais risco de sobreajuste); por isso só os exames com taxa de ausência acima de um
limiar recebem indicador (ver D4).

**C4. Ausente estrutural ≠ ausente por falta de dado.** `hemocultura_agente` vazio quando a hemocultura é negativa significa
"não se aplica", e não "dado perdido". Imputar mediana/moda aí destruiria a informação. Esses casos viram categorias
explícitas (`nao_se_aplica`, `nao_realizada`) antes do one-hot.

**C5. Escala.** KNN (distância), Regressão Logística e MLP (gradientes) dependem da escala; árvores (DT, RF, GB, XGBoost)
não. Um único pré-processador padronizado atende a todos sem prejudicar as árvores. Variáveis assimétricas (PCR, lactato,
leucócitos, TGO) recebem `log1p` antes da padronização para que poucos valores extremos não dominem a distância.

**C6. One-hot.** Categorias nominais (raça/cor, agente isolado) viram colunas 0/1; sem ordem artificial.
`handle_unknown="ignore"` evita erro quando o teste traz categoria que não existia no treino.

## 3. Decisões

| # | Decisão | Escolha | Motivo | Quem decidiu |
|---|---|---|---|---|
| P1 | Vazamento de alvo | Excluir sempre `dias_ate_dx_sepse`; as demais de `VARS_RISCO_VAZAMENTO` ficam com interruptor `excluir_vazamento` (padrão: manter) | `dias_ate_dx_sepse` é NaN se e somente se não houve sepse (prevê o alvo com 100% de acerto); as outras são preditoras previstas no PDF e permitem comparar com/sem | Usuário |
| P2 | Imputação | Mediana + indicador de ausência como padrão; `knn` e `iterativo` disponíveis por parâmetro para comparar depois | Linha de base simples, robusta a outliers e explicável | Usuário |
| P3 | Nome e local do plano | `plans/preprocessamento-plan.md` | Um arquivo por etapa do KDD | Usuário |
| P4 | Indicador de ausência só em exames com taxa de ausência no treino ≥ **20%** (`limiar_indicador`) | Parâmetro | Limita o aumento de colunas. *Alterado de 10% para 20% em 2026-10-28:* com 10% o protótipo gerava 32 indicadores (matriz de 153 colunas), porque as medidas `_pos24h` têm 10 p.p. a mais de ausentes por construção; com 20% são 11 (132 colunas) | Usuário (após ver o protótipo) |
| P5 | Divisão | 80/20 estratificada, `random_state` fixo, gravada em `data/treino.csv` e `data/teste.csv` | p. 45; reprodutível e auditável (pasta `data/` fora do git) | PDF |
| P6 | Pré-processador não ajustado + saída em `DataFrame` | `set_output(transform="pandas")` | Mantém nomes de colunas para o SHAP e evita vazamento (C1) | Proposta |
| P7 | Colapso das 97 preditoras (`_pre`/`_pos24h`) | **Não nesta etapa** | Decisão D4 do seed: colapsar só se necessário, depois de ver o desempenho | Usuário (seed) |
| P8 | Dependência nova | `scikit-learn` (1.9.1 instala no Python 3.14; verificado com `pip install --dry-run`) | Imputers, scalers, `ColumnTransformer`. `imbalanced-learn` e `xgboost` ficam para a etapa de modelos | Proposta |
| P9 | `id_paciente` e `sepse_tardia` | Fora das variáveis de entrada (`X`); alvo vira `y` | Identificador não é preditor | Proposta |
| P10 | `log1p` em exames de cauda longa | Lista **explícita** de 14 bases (28 colunas): leucócitos, neutrófilos, bastões, plaquetas, PCR, PCT, creatinina, bilirrubinas (3), lactato, TGO, TGP, relação neutrófilos/linfócitos | Usuário (aprovado); lista por conhecimento clínico e não calculada dos dados, para valer igual no dado real |
| P11 | Fora do `log1p` | Glicemia, hemoglobina, hematócrito, pH, Na, K, Ca ionizado (simétricos) e `relacao_it` (limitada a 0–1, assimetria 0,4) | Proposta |
| P12 | Colunas binárias | Detectadas no treino (valores ⊆ {0,1}, não vazias) e passam sem escala | Proposta: preserva a leitura "sim/não" no SHAP |
| P13 | One-hot | `OneHotEncoder(handle_unknown="ignore", drop="if_binary")`: `sexo` e `tipo_parto` viram 1 coluna | Proposta |
| P14 | Ausente em categórica fora da hemocultura | Vira a categoria `"ausente"` | Proposta (robustez para o dado real) |
| P15 | Ausente numérico fora dos exames | Mediana do treino | Proposta (no seed não ocorre; no dado real pode) |
| P16 | Passos sem aprendizado (hemocultura, `log1p`, conversão para `float`) | `FunctionTransformer` dentro do mesmo `Pipeline` | Proposta: nada é aprendido, então não vaza; o pré-processador é um objeto só |

## 4. Projeto

### 4.1 Grupos de colunas (definidos em constantes no módulo, não importados do gerador)

O módulo de pré-processamento deve funcionar com o CSV real do SOUL MV, então **não importa nada de `gerador_seed.py`**;
as listas de colunas são constantes próprias, a serem ajustadas quando o conector real existir.

| Grupo | Colunas | Tratamento |
|---|---|---|
| Identificador | `id_paciente` | Removido de `X` |
| Alvo | `sepse_tardia` | `y` |
| Excluída sempre | `dias_ate_dx_sepse` | Removida (P1) |
| Sinalizadas | `dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias`, `hemocultura_resultado`, `hemocultura_agente`, `perfil_sensibilidade` | Mantidas; removidas se `excluir_vazamento=True` |
| Binárias 0/1 | `vm_invasiva`, `cvc`, `npt`, `ecn`, `vasopressor`, flags de ATB, intercorrências, etc. | Passam sem alteração |
| Contagens e ordinais | `apgar_*`, `hiv_grau`, `n_gestacoes_anteriores`, `apneia_bradicardia_24h`, escores | Padronizadas |
| Contínuas sem ausentes | IG, peso, idade materna, sinais vitais, Glasgow | Padronizadas |
| Exames (44, com ausentes) | `*_pre`, `*_pos24h` | `log1p` nos assimétricos → imputação → (indicador se taxa ≥ limiar) → padronização |
| Categóricas nominais | `sexo`, `tipo_parto`, `raca_cor_rn`, `raca_cor_mae`, `colonizacao`, `hemocultura_*`, `perfil_sensibilidade` | Ausente estrutural vira categoria (C4) → one-hot |

### 4.2 Funções previstas em `preprocessamento.py`

- `carregar(caminho) -> DataFrame`: lê o CSV e valida colunas obrigatórias (erro claro se faltar alguma).
- `auditar(df) -> dict` / `imprimir_auditoria(...)`: nº de linhas, prevalência, ausentes por coluna, colunas constantes ou
  colunas com ≥ 95% no mesmo valor (no seed: eventos raros como `ecn` e `atb_prev_carbapenemicos`), violações dos
  critérios de inclusão (IG ≥ 22, `dias_utin` > 3),
  `id_paciente` duplicado, e uma checagem de vazamento (AUC de cada coluna sozinha; alerta se > 0,95).
- `tratar_ausentes_estruturais(X) -> X`: C4 (hemocultura); função pura, aplicada antes do pré-processador (não aprende nada
  dos dados, então não vaza).
- `dividir(df, teste=0.20, seed=42) -> X_treino, X_teste, y_treino, y_teste`: `train_test_split` estratificado.
- `construir_preprocessador(X_treino, estrategia="mediana", limiar_indicador=0.10, excluir_vazamento=False) -> ColumnTransformer`:
  **não ajustado**; só consulta `X_treino` para decidir *quais colunas* recebem indicador (taxa de ausência no treino, sem
  olhar o alvo; efeito desprezível sobre vazamento, documentado).
- CLI: `python preprocessamento.py --entrada data/seed_sepse.csv --seed 42` imprime a auditoria, grava `data/treino.csv` e
  `data/teste.csv` e mostra o formato da matriz transformada do treino.

### 4.3 Como as peças se encaixam (para as próximas etapas)

```
df ─► dividir ─► X_treino/y_treino ─┐
                 X_teste/y_teste ───┼─► (guardado, não usado até o fim)
construir_preprocessador ──► Pipeline([prep, SMOTE*, modelo]) ──► GridSearchCV (k=5)   (*etapa seguinte)
```

## 5. Passos de implementação (um por vez, com confirmação entre eles)

1. **Carregar e auditar**: `carregar`, `auditar`, relatório no terminal. Explicar C3/C4 com os números reais do seed.
2. **Dividir**: `dividir` + gravação em `data/`. Explicar C1/C2.
3. **Pré-processador**: `tratar_ausentes_estruturais` + `construir_preprocessador` (mediana). Explicar C3/C5/C6.
4. **Estratégias alternativas e interruptores**: `knn`, `iterativo`, `excluir_vazamento`, `limiar_indicador`.
5. **Fechamento**: CLI, testes, atualização de `plans/preprocessamento-plan.md` e commit.

### 5.1 Andamento

| Passo | Estado | Observações |
|---|---|---|
| 1. Carregar e auditar | **Concluído** (`preprocessamento.py`: `carregar`, `auditar`, `imprimir_auditoria`; 7 testes em `tests/test_preprocessamento.py`) | Seed 42: 1000 linhas, 97 preditoras, 20,0% de sepse, 0 ids duplicados, 0 violações de inclusão, 48 colunas com ausentes. `dias_ate_dx_sepse` detectada como vazamento (AUC 1,000, via indicador de ausência). Quase constantes (≥ 95%): `ecn` (96,4%) e `atb_prev_carbapenemicos` (96,2%). Maiores AUC isoladas: `pcr_pos24h` 0,899, `pct_pos24h` 0,862, `pcr_pre` 0,853, `news2` 0,825 |
| 2. Dividir | **Concluído** (`dividir`, `separar_x_y`, gravação de `data/treino.csv` e `data/teste.csv` pelo CLI; 4 testes) | 800 treino (160 positivos) / 200 teste (40 positivos), ambos com 20,0%; `X` com 96 variáveis (97 menos `dias_ate_dx_sepse`). **Desvio do desenho:** `dividir` devolve os dois DataFrames completos (com `id_paciente` e alvo) para a divisão ser gravada e auditável; a separação em X e y ficou numa função à parte, `separar_x_y`. `scikit-learn 1.9.1` instalado e adicionado ao `requirements.txt` |
| 3. Pré-processador | **Concluído** (`tratar_ausentes_estruturais`, `aplicar_log1p`, `construir_preprocessador`; 10 testes novos, 38 no total) | Matriz do treino **800 × 132**, 0 ausentes, 11 indicadores de ausência (21 binárias + 67 numéricas/exames + 11 indicadores + 33 one-hot). Detalhes em 5.2 |
| 4. Estratégias e interruptores | Pendente | |
| 5. Fechamento | Pendente | |

### 5.2 Passo 3: o que foi implementado

`construir_preprocessador(X_treino, limiar_indicador=0.20)` devolve um `Pipeline` do scikit-learn **sem ajuste**
(`set_output(transform="pandas")`, índice preservado para alinhar com `y`):

```
X ─► estrutural ─► log1p ─► ColumnTransformer ─────────────────────────────► float
     hemocultura   28 colunas  bin  passthrough (21 binárias)
                               num  mediana → padronizar (23 numéricas + 44 exames)
                               ind  MissingIndicator (11 exames com ausência ≥ 20%)
                               cat  "ausente" → one-hot (8 categóricas, 33 colunas)
```

- **Hemocultura** (`tratar_ausentes_estruturais`): resultado `negativa` → agente e sensibilidade `nao_se_aplica`;
  resultado ausente → `nao_realizada`; positiva sem agente → `nao_informado`.
- **`log1p`** (`aplicar_log1p`): levanta `ValueError` com valor ≤ −1 (ex.: código de ausente do sistema). Efeito no seed:
  assimetria do `pcr_pre` de 8,1 para perto de 1; do `pct_pre` de 4,0 para 1,6.
- **Só olha `X_treino` para decidir listas de colunas** (binárias; exames acima do limiar). Não usa o alvo e não aprende
  valores. Dentro da validação cruzada essas listas vêm do treino inteiro, e não de cada dobra: efeito desprezível.
- **CLI:** ajusta no treino e imprime `800 x 132 (11 indicadores de ausência), 0 ausentes`. Não transforma o teste.
- **Etapa final de conversão para `float`** (acrescentada durante a implementação): os indicadores saíam como `bool` e as
  binárias como `int64`; a matriz sai com um único tipo.
- **Testes** (10 novos): sem NaN, nomes únicos e índice preservado; colunas proibidas fora da saída; medianas calculadas só
  do treino e `transform` não altera o estado aprendido; hemocultura; indicadores só acima do limiar (e limiares 0 e 1);
  padronização e binárias intactas; `log1p` reduz assimetria e rejeita valor inválido; categoria nova no teste vira zeros;
  ausentes fora dos exames; pré-processador nasce sem ajuste e é clonável.

Diferenças em relação ao desenho aprovado: o limite de indicador é **20%** (P4) e a lista de `log1p` tem **14 bases**
(o rascunho dizia 13; contei errado, a lista inclui as três bilirrubinas e a relação neutrófilos/linfócitos).

## 6. Testes previstos (`tests/test_preprocessamento.py`)

- Divisão: 80/20, proporção de positivos preservada (±1 caso), sem `id_paciente` em comum, reprodutível com a mesma semente.
- `id_paciente`, `sepse_tardia` e `dias_ate_dx_sepse` ausentes das colunas transformadas.
- Transformação do treino e do teste sem nenhum NaN; nomes de colunas únicos.
- **Sem vazamento**: alterar valores do conjunto de teste e reajustar o pré-processador no treino não muda a transformação do treino;
  a mediana usada no teste é a do treino.
- Hemocultura: `negativa` → agente `nao_se_aplica`; resultado ausente → `nao_realizada`; nenhum NaN residual.
- Indicadores só para exames com taxa de ausência ≥ limiar (PCT sim; plaquetas não).
- Padronização: média ≈ 0 e desvio ≈ 1 nas contínuas do treino.
- Categoria nova no teste não gera erro (`handle_unknown="ignore"`).
- `excluir_vazamento=True` remove exatamente as colunas sinalizadas; `knn` e `iterativo` rodam (teste rápido).
- Auditoria: sinaliza `dias_ate_dx_sepse` como vazamento (AUC ≈ 1,0) e `ecn`/`atb_prev_carbapenemicos` como quase constantes.

## 7. Verificação ponta a ponta

1. `.venv\Scripts\python.exe -m pip install scikit-learn` e atualizar `requirements.txt`.
2. `.venv\Scripts\python.exe gerador_seed.py` (se faltar `data/seed_sepse.csv`) e depois `python preprocessamento.py`.
3. Conferir no relatório: 800/200, 160/40 positivos, matriz do treino sem NaN, nº de colunas esperado (~97 menos exclusões,
   mais one-hot, mais indicadores).
4. `.venv\Scripts\python.exe -m pytest` (seed + pré-processamento).

## 8. Riscos e pontos em aberto

- **O PDF não define o procedimento de imputação** (p. 40 remete a uma descrição que não existe no texto) e, na p. 38, manda
  excluir prontuários com variáveis incompletas. Propomos imputar; levar à orientadora.
- **Teste pequeno**: 200 linhas e ~40 positivos dão métricas com intervalo de confiança largo; a comparação entre modelos
  deve vir principalmente da validação cruzada no treino.
- **Seed MCAR**: o benefício do indicador de ausência só aparece com dados reais; aqui ele pode até piorar levemente.
- **Escores qSOFA/NEWS2 saturam, mas não são constantes:** no seed `qsofa` só vale 2 ou 3 e `news2` tem AUC isolada de 0,83
  (a versão inicial deste plano os chamava de "quase constantes"; a auditoria do passo 1 mostrou que não). Ficam como estão;
  a decisão de removê-los fica para depois de ver o desempenho.
- **Redundância:** `news2` e `qsofa` são calculados a partir de FR, PAS, FC, SpO2, temperatura e Glasgow, que também
  entram como colunas. Isso gera colinearidade (afeta RL e interpretação do SHAP); avaliar depois.
- **Colapso das colunas `_pre`/`_pos24h`** continua pendente (P7).

## 9. Histórico de mudanças

| Data | Mudança |
|---|---|
| 2026-10-28 | Plano aprovado (P1 a P9): excluir só `dias_ate_dx_sepse`; mediana + indicador; plano em `plans/` |
| 2026-10-28 | Esclarecido com o usuário: "retirar" uma variável = não entregá-la ao modelo (o CSV não muda); a "zona cinzenta" (`dias_utin`, `*_dias`, `hemocultura_*`) fica por padrão e o interruptor `excluir_vazamento` é só análise de sensibilidade (pode esperar ou ser descartado). Recomendação para os dados reais: pedir na extração do SOUL MV que os dias de internação e de dispositivos sejam contados **até o momento da suspeita** |
| 2026-10-28 | Passo 1 concluído. Correção: `qsofa`/`news2` não são "quase constantes" (qsofa só vale 2 ou 3; news2 tem AUC 0,83); textos corrigidos aqui e em `seed-plan.md` |
| 2026-10-28 | Passo 2 concluído: divisão 80/20 estratificada, `scikit-learn` adicionado, 11 testes novos (28 no total) |
| 2026-10-28 | Passo 3 concluído: `construir_preprocessador` (hemocultura, `log1p`, mediana, indicador de ausência ≥ 20%, escala, one-hot, saída em `float`); P4 alterado de 10% para 20%; P10 a P16 registrados; 10 testes novos (38 no total). README criado |
