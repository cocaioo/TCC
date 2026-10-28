# Plano de implementação: passo 1 (carregar e auditar) do pré-processamento

> **Reconstruído** em 2026-10-28 a partir do plano geral (`preprocessamento-plan.md`), do código e do commit `98d2266`.
> Não é o texto original aprovado: o passo foi aprovado dentro do plano geral, sem arquivo próprio. Os desvios e os
> números do seed são os que constam no plano geral (subseção 5.1).
> O antigo plano geral (`preprocessamento-plan.md`) e o `seed-plan.md` foram removidos de `plans/` em 2026-10-28, a pedido
> do usuário (um arquivo por passo); continuam no histórico do git, ex.: `git show 8cc96a9:plans/preprocessamento-plan.md`.

Próximo passo:
[`preprocessamento-passo2-plan.md`](preprocessamento-passo2-plan.md).

| | |
|---|---|
| **Status** | **Implementado** e commitado (`98d2266`, junto com o passo 2) |
| **Branch** | `create-pipeline` |
| **Código** | `preprocessamento.py` (`carregar`, `auditar`, `imprimir_auditoria`, auxiliar `_auc`), `tests/test_preprocessamento.py` (7 testes) |
| **Fonte no PDF** | "Auditoria e validação manual" (pp. 43–44); critérios de inclusão (p. 37) |

## 1. Contexto e objetivo

Antes de transformar qualquer coisa, é preciso saber o que chegou: quantas linhas, quantos ausentes, se há coluna
constante ou suspeita de vazamento, se os critérios de inclusão foram respeitados. Este passo é o equivalente automatizado
da auditoria manual prevista no PDF. **Nada aqui "aprende" dos dados nem altera o DataFrame:** só lê e relata.

Entrega: `carregar(caminho)` (lê o CSV e valida o mínimo), `auditar(df)` (devolve um dicionário com o resumo) e
`imprimir_auditoria(a)` (relatório no terminal).

Fora do escopo: dividir treino e teste (passo 2), qualquer transformação (passo 3).

## 2. Conceitos (para ler antes de aprovar)

**C3. Tipos de ausência.** MCAR (aleatória), MAR (depende de outra variável observada) e MNAR (depende do próprio valor).
No hospital real a ausência de PCT é informativa (pede-se PCT a quem parece mais grave); no seed a ausência é MCAR. A
auditoria mostra *quanto* falta em cada coluna, que é o insumo para decidir o tratamento (passo 3).

**C4. Ausente estrutural ≠ ausente por falta de dado.** `hemocultura_agente` vazio quando a hemocultura é negativa
significa "não se aplica", e não "dado perdido". A auditoria é onde isso aparece primeiro (taxas de ausência altas e
explicáveis); o tratamento vem no passo 3.

**Vazamento de alvo.** Uma coluna que prevê o desfecho quase perfeitamente só por existir ou por faltar é suspeita. No seed,
`dias_ate_dx_sepse` é NaN se e somente se não houve sepse. Por isso a auditoria calcula, para cada coluna numérica, a AUC do
valor sozinho e, havendo NaN, a AUC do indicador "está ausente"; acima de 0,95 gera alerta. Variáveis legítimas muito
informativas (ex.: PCR) ficam abaixo do limiar e não são acusadas.

## 3. Decisões

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P1 | Vazamento de alvo | `dias_ate_dx_sepse` sai sempre das variáveis de entrada (a exclusão em si é feita no passo 2, em `separar_x_y`); a auditoria apenas a **detecta** | Usuário |
| P3 | Nome e local do plano | Inicialmente `plans/preprocessamento-plan.md` (um arquivo por etapa do KDD); em 2026-10-28 passou a **um arquivo por passo** (`preprocessamento-passoN-plan.md`) | Usuário |
| P9 | `id_paciente` e `sepse_tardia` | São colunas obrigatórias: `carregar` falha com mensagem clara se faltarem; o alvo deve ser 0/1, sem ausentes | Proposta |
| — | Limiar de "quase constante" | Coluna com ≥ 95% das linhas no mesmo valor é listada (`LIMIAR_QUASE_CONSTANTE = 0.95`) | Proposta |
| — | Limiar de vazamento | AUC isolada > 0,95 gera alerta (`LIMIAR_AUC_VAZAMENTO = 0.95`) | Proposta |
| — | Critérios de inclusão | IG ≥ 22 semanas e internação > 72 h, i.e., `dias_utin` > 3 (`IG_MINIMA_SEMANAS`, `DIAS_UTIN_MINIMO`), conforme p. 37; violações são contadas, não corrigidas | PDF |

## 4. Desenho

`carregar(caminho)`: `pd.read_csv`; `ValueError` se faltar `id_paciente` ou `sepse_tardia`, ou se o alvo tiver valor fora de
{0, 1} ou ausente.

`auditar(df)` devolve um `dict` com:

| Chave | Conteúdo |
|---|---|
| `n_linhas`, `n_preditoras`, `n_positivos`, `prevalencia` | Tamanho e desfecho (preditoras = colunas menos id e alvo) |
| `ids_duplicados` | Pacientes repetidos |
| `violacoes_ig`, `violacoes_dias_utin` | Linhas fora dos critérios de inclusão (`None` se a coluna não existir) |
| `taxa_ausente` | Fração de NaN por coluna, só as que têm algum, em ordem decrescente |
| `quase_constantes` | Colunas com ≥ 95% no mesmo valor e a fração |
| `auc_por_coluna` | AUC (Mann-Whitney, `max(AUC, 1 − AUC)`) de cada coluna numérica sozinha, e do indicador de ausência quando há NaN |
| `suspeitas_vazamento` | Colunas com AUC > 0,95 |

`_auc` usa postos (sem dependência de scikit-learn). `imprimir_auditoria(a, top=10)` mostra o resumo, as 10 maiores taxas de
ausência, as quase constantes, as 10 maiores AUC e o alerta de vazamento.

O módulo **não importa nada de `gerador_seed.py`**: os grupos de colunas são constantes próprias, para funcionar igual com o
CSV real do SOUL MV.

## 5. Testes (7, em `tests/test_preprocessamento.py`)

1. `carregar` lê o CSV com o formato esperado.
2. `carregar` exige id e alvo.
3. `carregar` rejeita alvo que não seja 0/1.
4. `auditar` resume os dados (linhas, positivos, preditoras, ids duplicados, violações, taxas de ausência em ordem).
5. `auditar` sinaliza `dias_ate_dx_sepse` como vazamento estrutural e **não** acusa `pcr_pre`.
6. `auditar` acha violações dos critérios de inclusão (IG < 22, `dias_utin` ≤ 3) e id duplicado.
7. `auditar` acha colunas quase constantes.

## 6. Verificação e resultado no seed 42

`.venv\Scripts\python.exe preprocessamento.py --entrada data/seed_sepse.csv --seed 42` imprime o relatório. Resultado
registrado no plano geral: 1000 linhas, 97 preditoras, 20,0% de sepse, 0 ids duplicados, 0 violações de inclusão, 48 colunas
com ausentes. `dias_ate_dx_sepse` detectada como vazamento (AUC 1,000, via indicador de ausência). Quase constantes:
`ecn` (96,4%) e `atb_prev_carbapenemicos` (96,2%). Maiores AUC isoladas: `pcr_pos24h` 0,899, `pct_pos24h` 0,862,
`pcr_pre` 0,853, `news2` 0,825.

## 7. Riscos e observações

- **Correção feita durante o passo:** o plano chamava `qsofa` e `news2` de "quase constantes"; a auditoria mostrou que não
  (qsofa só vale 2 ou 3; `news2` tem AUC de 0,83). Eles *saturam*, mas não são constantes. Textos corrigidos no plano geral e
  em `seed-plan.md`.
- **AUC isolada não detecta tudo:** só pega vazamento forte e de uma coluna; vazamento combinado ou temporal (ex.: dias de
  dispositivo contados até o fim da internação) exige análise clínica, não estatística. É a "zona cinzenta" do plano geral.
- **A auditoria só cobre colunas numéricas** para a AUC; as categóricas entram apenas nas taxas de ausência e nas quase
  constantes.
