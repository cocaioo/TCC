# Plano de implementação: passo 4 (estratégias de imputação e interruptor) do pré-processamento

Plano do passo 4. Referências a "plano geral" são ao antigo `preprocessamento-plan.md`, removido de `plans/` em 2026-10-28
(continua no histórico do git, ex.: `git show 8cc96a9:plans/preprocessamento-plan.md`); passo anterior:
[`preprocessamento-passo3-plan.md`](preprocessamento-passo3-plan.md).

| | |
|---|---|
| **Status** | Plano escrito em 2026-10-28. **Escopo reduzido em 2026-10-29 (aprovado pelo usuário)**; nada implementado neste passo |
| **Branch** | `create-pipeline` |
| **Código previsto** | Original: `preprocessamento.py`, `tests/test_preprocessamento.py`. Após a redução: só o interruptor `excluir_sinalizadas`, a ser feito na etapa de modelos |
| **Escopo escolhido** | Original: completo + experimento de recuperação, com o iterativo incluído (usuário). Ver a seção 0 para a revisão |

## 0. Revisão de escopo (2026-10-29, após a revisão conceitual)

> Este plano foi escrito antes da revisão conceitual (SMOTE, GridSearchCV, validação cruzada). As seções 1 a 9 abaixo
> são o **plano original, mantido para rastreabilidade**; o que vale é esta seção, **aprovada pelo usuário em 2026-10-29**.

O passo 4 tinha três partes (a `estrategia`, com duas variantes, o experimento `comparar_imputacao` e o interruptor
`excluir_sinalizadas`). Avaliadas à luz do restante do pipeline, só o interruptor tem justificativa própria:

| Parte | O que faz | Proposta | Motivo |
|---|---|---|---|
| `estrategia` = `knn` | Imputação pelos vizinhos mais próximos | **Adiar**, como possível botão do GridSearchCV depois que o pipeline rodar de ponta a ponta | A pergunta "qual imputação é melhor?" é respondida de forma mais direta pelo F1 do modelo final. O KNN é barato (0,3 s por ajuste) |
| `estrategia` = `iterativo` | Imputação por regressões repetidas | **Descartar** | 16,6 s por ajuste, com `ConvergenceWarning`; inviável dentro de uma busca em grade (5 dobras × combinações). O próprio plano original já previa descartá-lo se não houvesse ganho claro |
| `comparar_imputacao` | Esconde valores conhecidos e mede o erro de recuperação | **Descartar** | Mede uma métrica "interna" da imputação, sobreposta pelo GridSearchCV. Com dado sintético e ausência aleatória (MCAR), o resultado não informa nada sobre o dado real |
| `excluir_sinalizadas` | Treina sem a "zona cinzenta" (`dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias`, `hemocultura_*`) | **Manter, mas na etapa de modelos**, como opção do pipeline (análise de sensibilidade) | Não se sobrepõe a nada: responde se o desempenho depende de variáveis que podem ser consequência da sepse. É barato. Detalhes de desenho nas seções 3 (C9) e 4 (P20) abaixo |

**Consequência para o pré-processamento:** o pré-processador do passo 3 (mediana, padrão inalterado) é a versão final desta
etapa; o passo 5 (fechamento) fica só com a documentação. As decisões P17 a P19, P21 a P24 e os testes 2, 3 e 6 da seção 6
**não serão implementados** (revisão aprovada).


## 1. Contexto

Os passos 1 a 3 estão prontos e commitados (`8cc96a9`): o pré-processador atual imputa os ausentes numéricos pela
**mediana** e deixa a "zona cinzenta" (`dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias`, `hemocultura_*`) como preditoras.
O PDF não define o procedimento de imputação (p. 40 remete a uma descrição que não existe no texto), então este passo
entrega as **alternativas** e uma forma de **julgar** entre elas, sem mexer no padrão atual:

1. `estrategia`: mediana (padrão, inalterada), `knn` ou `iterativo`.
2. `excluir_sinalizadas`: interruptor para treinar sem a zona cinzenta (análise de sensibilidade).
3. `comparar_imputacao`: experimento que esconde valores conhecidos do treino e mede o erro de cada estratégia.

Fora do escopo: modelos, SMOTE, GridSearchCV; ajustar hiperparâmetros dos imputadores (valores iniciais são palpites
"a revisar"); usar o alvo em qualquer imputação.

## 2. Medições que embasam o desenho (seed 42, treino 800 × 67 numéricas)

| Fato | Valor | Consequência |
|---|---|---|
| `KNNImputer(k=5)` | 0,27 s por ajuste | Viável dentro de GridSearchCV |
| `IterativeImputer(max_iter=10)` | **16,6 s** por ajuste, com `ConvergenceWarning` | 5 dobras × combinações de hiperparâmetros fica lento; documentar |
| Correlação `_pre` × `_pos24h` | PCR 0,89; plaquetas 0,93 | KNN e iterativo têm o que aprender (a mediana ignora isso) |
| Sem as 7 sinalizadas | 132 → **110** colunas (4 numéricas + 18 colunas one-hot da hemocultura) | Valor de referência para o teste do interruptor |
| `IterativeImputer` | exige `from sklearn.experimental import enable_iterative_imputer` | Import no topo do módulo |

## 3. Conceitos (para ler antes de aprovar)

**C7. Imputação por vizinhos e iterativa.** A mediana dá o mesmo número para todo mundo. O KNN procura os 5 pacientes do
**treino** mais parecidos (nas colunas que ambos têm) e usa a média deles; o iterativo faz uma regressão de cada coluna com
ausentes sobre as demais, repetidamente. Ambos aproveitam relações entre colunas (ex.: PCR pré e pós-24h). Duas
consequências: (a) distância exige **escala**, então nesses dois a padronização vem *antes* da imputação (o `StandardScaler`
ignora NaN ao ajustar); (b) sem vazamento: o que é "aprendido" (os doadores, as regressões) vem só do treino; ao
transformar o teste, cada linha busca doadores no treino e **nunca em outras linhas do teste**. O alvo nunca entra na
imputação (se entrasse, o rótulo vazaria para dentro das variáveis).

**C8. Como julgar uma imputação.** No dado real nunca vemos o valor que faltou, então não dá para medir o erro direto. O
método padrão é o **experimento de recuperação**: esconder de propósito 10% dos valores que *temos*, imputar e comparar com o
valor verdadeiro (RMSE em desvios-padrão; 1,0 é o erro de "chutar a mediana"). Limites: o mascaramento é aleatório (MCAR);
no dado real, com ausência informativa (MNAR), a vantagem pode ser outra, e o indicador de ausência (P4 do passo 3) segue
cobrindo esse caso. A comparação por F1 do modelo fica para a etapa de modelos.

**C9. O que o interruptor responde.** `excluir_sinalizadas=True` não "corrige" vazamento: serve para ver quanto do
desempenho depende das variáveis que podem ser consequência da sepse. Se o F1 cair muito sem elas, é sinal de que o modelo
usava a consequência; é argumento para levar à orientadora (e para pedir os dias contados até a suspeita).

## 4. Decisões

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P17 | Parâmetro `estrategia` em `construir_preprocessador` | `"mediana"` (padrão), `"knn"`, `"iterativo"`; outro valor levanta `ValueError` | Usuário (escopo) |
| P18 | Ordem no ramo numérico | Mediana **inalterada** (imputar → escalar; preserva o passo 3 e seus testes). `knn` e `iterativo`: escalar → imputar | Proposta (C7) |
| P19 | Hiperparâmetros iniciais | `KNNImputer(n_neighbors=5)`; `IterativeImputer(max_iter=10, random_state=0)` (estimador padrão, `BayesianRidge`). Palpites, não ajustados | Proposta, "a revisar" |
| P20 | Interruptor, com nome novo | `excluir_sinalizadas` (padrão `False`). Substitui o nome `excluir_vazamento` do plano geral (P1): as colunas são **suspeitas**, não vazamento comprovado. Implementação por omissão: as 7 colunas ficam fora das listas de cada ramo e o `ColumnTransformer` (`remainder="drop"`) as descarta | Proposta |
| P21 | Imputação só nas 67 colunas numéricas | Binárias e categóricas não entram como "informantes" do KNN/iterativo | Proposta (simplicidade); limitação registrada |
| P22 | Experimento `comparar_imputacao` | Padroniza as numéricas do treino (após estrutural e `log1p`), esconde 10% dos valores observados (semente fixa), imputa com cada estratégia e devolve RMSE geral, RMSE só nos exames, razão sobre a mediana e tempo | Usuário (escopo) |
| P23 | Fábrica única de imputadores | `_imputador(estrategia)` é usada pelo pipeline e pelo experimento, para os dois não divergirem | Proposta |
| P24 | `ConvergenceWarning` do iterativo | Não é suprimido: é informação para decidir se o iterativo vale a pena | Proposta |

## 5. Código a escrever (`preprocessamento.py`)

- Constantes: `ESTRATEGIAS = ("mediana", "knn", "iterativo")` e `FRACAO_MASCARA_PADRAO = 0.10`.
- Imports: `KNNImputer`, `enable_iterative_imputer` e `IterativeImputer`.
- `_imputador(estrategia)`: devolve `SimpleImputer(median)`, `KNNImputer` ou `IterativeImputer`; valida o nome.
- `construir_preprocessador(X_treino, limiar_indicador=0.20, estrategia="mediana", excluir_sinalizadas=False)`:
  - com `excluir_sinalizadas`, tira `COLS_SINALIZADAS` de `categoricas`, `binarias` e `numericas`;
  - ramo `num`: `[imputar, escalar]` para a mediana e `[escalar, imputar]` para as outras (nomes de passo iguais).
- `comparar_imputacao(X_treino, fracao=0.10, seed=42, estrategias=ESTRATEGIAS) -> DataFrame`: usa
  `tratar_ausentes_estruturais`, `aplicar_log1p` e as 67 colunas numéricas; não altera `X_treino`; lê só o treino.
- CLI: `--estrategia {mediana,knn,iterativo}`, `--excluir-sinalizadas` e `--comparar-imputacao` (imprime a tabela do
  experimento; leva cerca de 17 s por causa do iterativo). O formato da matriz impresso reflete as opções.
- Cabeçalho do módulo: acrescentar "Passo 4". Sem arquivos novos de código e sem dependências novas.

## 6. Testes (`tests/test_preprocessamento.py`, de 38 para cerca de 45)

1. `estrategia` desconhecida levanta `ValueError`.
2. `knn` e `iterativo`: sem NaN, mesmas colunas, mesmo índice e nomes únicos que a mediana (iterativo num subconjunto de
   cerca de 200 linhas, para ficar rápido).
3. **Sem vazamento, linha a linha:** `transform(X_teste.iloc[:50])` é igual às 50 primeiras linhas de `transform(X_teste)`
   (vale para `knn` e `iterativo`; prova que o teste não usa o próprio teste como doador). Valores extremos no teste não
   mudam a transformação do treino.
4. Padrão inalterado: `estrategia="mediana"` dá 800 × 132 e os 10 testes do passo 3 continuam passando.
5. `excluir_sinalizadas=True`: saem exatamente as 4 numéricas e as colunas one-hot da hemocultura (132 → 110); as binárias
   `cvc`, `vm_invasiva` e `npt` permanecem; a diferença de colunas é só essa.
6. `comparar_imputacao` (mediana e KNN): RMSE finito; KNN < mediana no seed (correlação de 0,9 entre janelas); mesma semente
   dá o mesmo resultado; `X_treino` intacto; o número de valores escondidos é cerca de 10% dos observados.

## 7. Verificação ponta a ponta

1. `.venv\Scripts\python.exe -m pytest -q`: os 38 anteriores mais cerca de 7 novos, todos passando.
2. `preprocessamento.py` sem opções: continua `800 x 132` (regressão do padrão).
3. `--estrategia knn` e `--estrategia iterativo`: `800 x 132`, 0 ausentes.
4. `--excluir-sinalizadas`: `800 x 110`.
5. `--comparar-imputacao`: tabela com RMSE por estratégia; conferir que a mediana fica perto de 1,0 e que o KNN é menor nos
   exames pareados (`_pre` × `_pos24h`).

## 8. Documentação a atualizar ao fim do passo

Este arquivo (status, o que foi implementado, desvios e o resultado do experimento) e o `README.md` (passo 4 → Feito e
comandos novos em "Como usar").
Commit só depois da confirmação (passo 5 = fechamento).

## 9. Riscos e observações

- **O seed é MCAR e as relações entre exames foram plantadas por mim:** o ganho do KNN e do iterativo aqui mostra que o
  código funciona e dá a ordem de grandeza; não prevê o resultado no dado real.
- **Custo do iterativo** (16,6 s por ajuste) o torna inviável em GridSearchCV completo; se o experimento não mostrar ganho
  claro sobre o KNN, a recomendação é descartá-lo antes da etapa de modelos.
- **`ConvergenceWarning`:** com `max_iter=10` o iterativo não converge no seed; `tol` e `max_iter` não serão ajustados agora.
- **Hiperparâmetros** (k = 5, estimador) são palpites; ajustá-los dentro da validação cruzada é tema da etapa de modelos.
- **Binárias fora da imputação** (P21): o KNN não usa `cvc`, `vm_invasiva` etc. para achar vizinhos; limitação aceita.
- **Excluir as sinalizadas não remove as flags** `cvc`, `vm_invasiva` e `npt` (exposição sim/não), que ficam.
