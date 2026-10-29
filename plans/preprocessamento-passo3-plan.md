# Plano de implementação: passo 3 (pré-processador) da etapa de pré-processamento

Plano do passo 3, como foi aprovado, com os desvios da implementação no fim. Referências a "plano geral" são ao antigo
`preprocessamento-plan.md`, removido de `plans/` em 2026-10-28 (um arquivo por passo); continua no histórico do git, ex.:
`git show 8cc96a9:plans/preprocessamento-plan.md`.

| | |
|---|---|
| **Status** | **Implementado** e commitado (`8cc96a9`) |
| **Branch** | `create-pipeline` |
| **Código** | `preprocessamento.py` (`tratar_ausentes_estruturais`, `aplicar_log1p`, `construir_preprocessador`), `tests/test_preprocessamento.py` |

> **Revisão de 2026-10-29 (adversarial, antes do commit dos passos de modelos):** a decisão de QUAIS colunas são binárias e
> QUAIS exames recebem o indicador "foi medido?" (P4, P12) deixou de ser tomada em `construir_preprocessador` com o treino
> inteiro e passou para o `fit` (seletores de coluna no `ColumnTransformer`). Motivo: na validação cruzada, a dobra de
> validação influenciava a estrutura do modelo. `X_treino` agora só fornece nomes de colunas. Resultado no seed inalterado
> (800 × 132, 11 indicadores). Teste novo em `tests/test_preprocessamento.py`.

## 1. Contexto

Os passos 1 (auditar) e 2 (dividir 80/20) estão prontos e commitados (`98d2266`). Falta transformar `X_treino` em uma matriz
numérica sem ausentes, que os oito algoritmos do Quadro 4 aceitem, **sem vazamento**: tudo o que for "aprendido" (medianas,
médias, desvios, categorias) vem só do treino (conceito C1 do plano geral).

Entrega: `construir_preprocessador(X_treino, limiar_indicador=0.20)` em `preprocessamento.py`, que devolve um
`sklearn.pipeline.Pipeline` **não ajustado**. Quem o ajusta é o treino (e, depois, cada dobra da validação cruzada).

Fora do escopo deste passo: KNN/iterativo e interruptor da zona cinzenta (passo 4), SMOTE e modelos (etapas seguintes),
gravar o objeto ajustado em disco.

## 2. Decisões

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P2 | Imputação | Mediana do treino | Usuário (aprovado) |
| P4 | Indicador "foi medido?" | Só para exames com ausência ≥ **20%** no treino (`limiar_indicador`), e não 10% | Usuário, após ver o protótipo: 10% geraria 32 indicadores (153 colunas); 20% gera 11 (132 colunas) |
| P6 | Saída | `DataFrame` com nomes de coluna (`set_output(transform="pandas")`), mesmo índice da entrada | Plano |
| P10 | `log1p` em exames de cauda longa | Sim, nesta versão. Lista **explícita** de 13 exames (26 colunas): leucócitos, neutrófilos, bastões, plaquetas, PCR, PCT, creatinina, bilirrubinas (3), lactato, TGO, TGP, relação neutrófilos/linfócitos | Usuário (aprovado); lista por conhecimento clínico, não por dado, para funcionar igual no dado real |
| P11 | Fora do `log1p` | Glicemia, hemoglobina, hematócrito, pH, Na, K, Ca ionizado (simétricos) e `relacao_it` (limitada a 0–1; assimetria 0,4) | Proposta |
| P12 | Colunas binárias | Detectadas no treino (valores ⊆ {0,1}) e passam **sem escala** | Proposta: preserva a interpretação (SHAP em "sim/não") |
| P13 | One-hot | `OneHotEncoder(handle_unknown="ignore", drop="if_binary")`: colunas de 2 categorias (`sexo`, `tipo_parto`) viram 1 coluna | Proposta |
| P14 | Ausente em coluna categórica fora da hemocultura | Vira a categoria `"ausente"` (robustez para o dado real) | Proposta |
| P15 | Ausente numérico fora dos exames | Mediana também (no seed não há; no dado real pode haver) | Proposta |
| P16 | Passos sem aprendizado (hemocultura e `log1p`) | `FunctionTransformer` dentro do mesmo `Pipeline`, antes do `ColumnTransformer` | Proposta: nada aprendido, então não vaza; o pré-processador é um objeto só |

## 3. Desenho

Ordem dentro do `Pipeline`:

```
X ─► [1] estrutural ─► [2] log1p ─► [3] ColumnTransformer ─► matriz (DataFrame)
      hemocultura       26 exames     ┌ bin    passthrough (21 binárias)
      "nao_se_aplica"   (sem ajuste)  ├ num    mediana → padronizar (23 numéricas + 44 exames)
      "nao_realizada"                 ├ ind    MissingIndicator (exames com ausência ≥ 20%)
                                      └ cat    "ausente" → one-hot (8 categóricas)
```

**[1] `tratar_ausentes_estruturais(X)`** (função pura, não aprende nada): em `hemocultura_agente` e `perfil_sensibilidade`:
resultado `negativa` → `nao_se_aplica`; resultado ausente → `nao_realizada`; resultado `positiva` sem agente →
`nao_informado`. Em `hemocultura_resultado`: ausente → `nao_realizada`. Conceito C4.

**[2] `aplicar_log1p(X)`** (função pura): `log1p` nas colunas `<base>_pre` e `<base>_pos24h` das bases de P10. Levanta
`ValueError` se houver valor ≤ −1 (dado inválido, ex.: código "−999" de ausente do sistema). NaN passa e é imputado depois.
Efeito medido no protótipo: a assimetria do `pcr_pre` cai de 8,1 para perto de 1 e a do `pct_pre` de 4,0 para 1,6.

**[3] `ColumnTransformer`** (`verbose_feature_names_out=False`, para nomes legíveis):

| Ramo | Colunas (no seed) | Transformação |
|---|---|---|
| `bin` | 21 binárias | `passthrough` |
| `num` | 23 contagens/contínuas + 44 exames | `SimpleImputer(median)` → `StandardScaler` |
| `ind` | exames com ausência ≥ `limiar_indicador` no treino (11) | `MissingIndicator(features="all")` → colunas `missingindicator_<exame>` |
| `cat` | 8 categóricas | `SimpleImputer(constant="ausente")` → `OneHotEncoder` |

`construir_preprocessador(X_treino, ...)` só **olha** `X_treino` para decidir listas de colunas (quais são binárias e quais
exames passam do limiar de ausência). Não olha o alvo, não aprende valores. Dentro da validação cruzada essas listas vêm do
treino completo (e não de cada dobra): efeito desprezível, documentado.

Resultado esperado no seed: **800 × 132** (21 + 67 + 11 + 33), sem NaN, nomes únicos.

## 4. Código a escrever

Em `preprocessamento.py` (novas constantes e funções, estilo igual ao dos passos 1 e 2):

- Constantes: `COLS_CATEGORICAS`, `SUFIXOS_EXAME = ("_pre", "_pos24h")`, `BASES_LOG1P`, `LIMIAR_INDICADOR_PADRAO = 0.20`.
- `tratar_ausentes_estruturais(X)`, `aplicar_log1p(X)`, `construir_preprocessador(X_treino, limiar_indicador=0.20)`.
- CLI: depois da divisão, `construir_preprocessador(...).fit_transform(X_treino)` e imprime o formato da matriz. **Não**
  transforma o teste no CLI (o teste só é aberto no fim; os testes automáticos é que verificam o `transform` dele).
- Cabeçalho do módulo: acrescentar "Passo 3".

Sem arquivos novos de código; sem novas dependências (`scikit-learn` já instalado).

## 5. Testes (`tests/test_preprocessamento.py`)

1. Treino e teste transformados sem NaN; nomes de colunas únicos; mesmo índice de entrada (alinhamento com `y`).
2. `id_paciente`, `sepse_tardia` e `dias_ate_dx_sepse` ausentes da saída.
3. **Sem vazamento:** ajustar o pré-processador com um treino fixo e alterar drasticamente os valores do teste não muda a
   transformação do treino; a mediana usada para imputar o teste é a do treino.
4. Hemocultura: `negativa` → `nao_se_aplica`; resultado ausente → `nao_realizada`; positiva sem agente → `nao_informado`;
   nenhum NaN residual.
5. Indicadores só nos exames com ausência ≥ limiar (PCT sim, plaquetas não); `limiar_indicador=1.0` não cria nenhum.
6. Padronização: média ≈ 0 e desvio ≈ 1 nas colunas numéricas e de exames do treino; binárias inalteradas (0/1).
7. `log1p` reduz a assimetria de PCR e PCT; levanta erro com valor ≤ −1.
8. Categoria nova no teste (ex.: `raca_cor_rn="nova"`) não gera erro e vira zero em todas as colunas daquela variável.
9. Robustez: NaN numérico fora dos exames (ex.: `idade_materna`) e NaN em categórica (`raca_cor_mae`) são tratados.
10. `construir_preprocessador` não está ajustado ao ser devolvido (usar antes do `fit` falha) e é reajustável (`clone` funciona).

## 6. Verificação ponta a ponta

1. `.venv\Scripts\python.exe -m pytest -q`: os 28 testes anteriores mais cerca de 10 novos, todos passando.
2. `.venv\Scripts\python.exe preprocessamento.py`: o relatório termina com "matriz do treino: 800 × 132, 0 ausentes".
3. Conferir no console que `pct_pre` está padronizado, que `missingindicator_pct_pre` existe e que `plaquetas_pre` não
   tem indicador.

## 7. Riscos e observações

- **Indicadores no seed são ruído** (ausência MCAR). A utilidade só aparece no dado real; por isso um limite mais alto é prudente.
- **132 colunas para ~160 positivos** continua alto; seleção/colapso de variáveis segue pendente (P7 do plano geral).
- **A lista de `log1p` e o conjunto de categóricas são constantes do seed**: precisam ser revistos quando chegar a extração
  real do SOUL MV (nomes, unidades e categorias podem ser diferentes).
- **Padronizar o escore NEWS2/qSOFA junto com seus componentes** gera colinearidade (afeta RL e a leitura do SHAP); observar depois.

## 8. Desvios entre o plano aprovado e a implementação

- A lista de `log1p` tem **14 bases (28 colunas)**, e não 13 (26): o rascunho contou errado; a lista inclui as três
  bilirrubinas e a relação neutrófilos/linfócitos.
- Acrescentada uma etapa final `_para_float` no `Pipeline`: os indicadores saíam como `bool` e as binárias como `int64`;
  a matriz agora sai com um único tipo.
- O CLI imprime também o número de indicadores de ausência.
- Resultado final conforme o esperado: 800 × 132, 0 ausentes, 11 indicadores; 38 testes passando no total.
