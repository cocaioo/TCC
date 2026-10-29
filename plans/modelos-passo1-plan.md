# Plano de implementação: passo 1 (catálogo de modelos e pipelines) da etapa de modelos

Primeiro passo da etapa 3 do README (mineração de dados). Depende do pré-processador do passo 3
([`preprocessamento-passo3-plan.md`](preprocessamento-passo3-plan.md)); o passo 4 do pré-processamento foi reduzido
([seção 0 do plano](preprocessamento-passo4-plan.md)). Conceitos de apoio: capítulos 4, 5 e 7 de `docs/material-de-estudo.pdf`.

| | |
|---|---|
| **Status** | Plano escrito e **aprovado** em 2026-10-29 (P6, P7 e grades confirmados); **implementado** (ver seção 11) |
| **Branch** | `create-pipeline` |
| **Código** | `modelos.py` (novo), `tests/test_modelos.py` (novo) |
| **Dependências** | `xgboost` e `imbalanced-learn` já instaladas e no `requirements.txt` (`1603bc4`) |

## 1. Contexto e escopo

O pré-processador entrega uma matriz numérica, sem ausentes (800 × 132 no seed). Falta o que vem depois dele: **quais
algoritmos** e **quais valores de hiperparâmetros** serão testados, e **como encaixá-los** em um `Pipeline` para a validação
cruzada refazer tudo a cada dobra, sem vazamento.

Este passo entrega só o **catálogo** (os 8 algoritmos do Quadro 4 do PDF, cada um com duas grades de hiperparâmetros) e a
função que monta o `Pipeline` de cada um. **Não roda GridSearchCV nem validação cruzada**: isso é o passo 2. Assim, o que
se discute aqui é só a configuração, que é o que você disse ser o mais importante para quando os dados reais chegarem.

Fora do escopo: busca em grade e comparação (passo 2); SMOTE e `excluir_sinalizadas` (passo 3); avaliação no teste e curvas
ROC (passo 4); calibração (proposta a validar com a orientadora); SVM e redes profundas (ver P7).

## 2. O que o PDF pede (base das decisões)

| Item do PDF | Onde | Consequência |
|---|---|---|
| 8 algoritmos: GB, XGBoost, DT, KNN, RF, NB, RL, ANN/MLP | Quadro 4 | O catálogo tem exatamente esses 8 |
| SVM com núcleos complexos e redes profundas "também serão avaliados, sendo excluídos caso apresentem resultados abaixo de 70% nas métricas" | Quadro 4, linha da ANN/MLP | Ficam fora do catálogo inicial (P7) |
| `GridSearchCV` para os melhores parâmetros | p. 45 | O catálogo expõe a grade de cada modelo no formato do `GridSearchCV` |
| k-fold com k = 5 e `cross_val_score`, métricas = médias das 5 rodadas | p. 46 | Passo 2 |
| Dividir 80/20 com `train_test_split` | p. 45 | Já feito (passo 2 do pré-processamento) |
| "Reamostragem sintética e ajuste de pesos de classe" serão avaliados | p. 47 | Os pipelines aceitam o modo de desbalanceamento (P4) |
| F1 é a métrica principal; também acurácia, precisão, recall e AUROC | Quadro 5 | Passo 2 |

## 3. Decisões

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P1 | Estrutura | Um módulo `modelos.py` (sem notebook), no mesmo estilo de `preprocessamento.py` | Preferência do usuário |
| P2 | Duas grades por modelo, escolhidas por parâmetro: `"enxuta"` (testar o pipeline, rápida) e `"completa"` (dado real) | Mantidas as duas no código, para não reescrever quando os dados reais chegarem | Usuário (conversa de 2026-10-29) |
| P3 | Valores das grades | Os das tabelas da seção 5, como **valores iniciais "a revisar"**; não são ajustados ao seed | Proposta |
| P4 | Desbalanceamento | `construir_pipeline(..., desbalanceamento="nenhum" \| "class_weight")` neste passo; `"smote"` entra no passo 3. Valor desconhecido levanta `ValueError` | Proposta; começar com `class_weight` (usuário) |
| P5 | Como aplicar `class_weight` | Modelos que aceitam o parâmetro (RL, DT, RF) recebem `class_weight="balanced"`; GB e XGBoost usam subtipos que calculam o peso a cada `fit` (ver P6 e a atualização de 2026-10-29, seção 11; o plano original usava `scale_pos_weight` com a prevalência do treino inteiro); **KNN, NB e MLP não têm esse recurso no scikit-learn e rodam sem ajuste**, o que fica registrado | Proposta (limite do scikit-learn, ver capítulo 5.2 do material) |
| P6 | Gradient Boosting com pesos | Subclasse mínima `GradientBoostingBalanceado` cujo `fit` calcula `compute_sample_weight("balanced", y)` e o repassa. Sem parâmetro novo no `__init__`, então `clone` e o `GridSearchCV` funcionam normalmente (cerca de 6 linhas) | Proposta. Alternativa: GB sem ajuste neste modo (mais simples, mas deixa o GB em desvantagem na comparação) |
| P7 | SVM e redes profundas do PDF | **Fora do catálogo inicial.** Entram depois dos 8 rodarem de ponta a ponta, com a regra de exclusão por resultado < 70% | Proposta; **precisa da sua confirmação** |
| P8 | Reprodutibilidade | `random_state=42` em todo estimador que o aceite; um parâmetro `seed` | Proposta |
| P9 | Prefixo dos hiperparâmetros | O pipeline tem dois passos, `prep` e `modelo`; as grades usam `modelo__<parâmetro>` | Convenção do scikit-learn |
| P10 | Regressão Logística | Só `C` na grade (regularização L2, o padrão). **Não** se usa o parâmetro `penalty`: foi descontinuado nas versões recentes do scikit-learn (instalada: 1.9.1) | Proposta |
| P11 | MLP | `max_iter=500` e `early_stopping=True` fixos, para não depender de aviso de convergência | Proposta |
| P12 | Pré-processador | O mesmo `construir_preprocessador(X_treino)` para todos os modelos, como **primeiro passo** do pipeline (é isso que garante o refazer por dobra) | Plano do passo 3 |

## 4. Desenho

```
construir_pipeline(preprocessador, estimador, desbalanceamento)
   -> Pipeline([("prep", preprocessador), ("modelo", estimador)])

catalogo(grade="enxuta", desbalanceamento="nenhum", seed=42)
   -> { "RL": (estimador, {"modelo__C": [...]}), "NB": (...), ... }   # 8 entradas
```

- Nenhum valor vem dos dados: os pesos de classe de GB e XGBoost são calculados no `fit`, com o `y` de cada ajuste (o plano
  original tinha um argumento `prevalencia`, removido em 2026-10-29).
- O catálogo devolve estimadores **sem ajuste**; quem ajusta é o passo 2 (GridSearchCV).
- `python modelos.py --listar` imprime o catálogo: modelo, nº de combinações em cada grade, e se aceita `class_weight`. Serve
  para você enxergar a configuração sem rodar nada.

## 5. Grades iniciais (valores "a revisar")

Combinações da grade **enxuta**: 59 no total. Com 5 dobras, são 295 ajustes (e 1475 se a avaliação for aninhada, ver seção 8).

| Modelo | Estimador | Grade enxuta | Combinações |
|---|---|---|---|
| RL | `LogisticRegression(max_iter=2000)` | `C` ∈ {0,01; 0,1; 1; 10} | 4 |
| NB | `GaussianNB` | `var_smoothing` ∈ {1e-9; 1e-8; 1e-7} | 3 |
| KNN | `KNeighborsClassifier` | `n_neighbors` ∈ {3; 5; 11; 21} × `weights` ∈ {uniform; distance} | 8 |
| DT | `DecisionTreeClassifier` | `max_depth` ∈ {3; 5; 8; None} × `min_samples_leaf` ∈ {1; 5; 10} | 12 |
| RF | `RandomForestClassifier` | `n_estimators` ∈ {100; 300} × `max_depth` ∈ {None; 8} × `min_samples_leaf` ∈ {1; 5} | 8 |
| GB | `GradientBoostingClassifier` | `n_estimators` ∈ {100; 200} × `learning_rate` ∈ {0,05; 0,1} × `max_depth` ∈ {2; 3} | 8 |
| XGBoost | `XGBClassifier(eval_metric="logloss")` | `n_estimators` ∈ {100; 300} × `learning_rate` ∈ {0,05; 0,1} × `max_depth` ∈ {3; 5} | 8 |
| MLP | `MLPClassifier(max_iter=500, early_stopping=True)` | `hidden_layer_sizes` ∈ {(32,); (64, 32)} × `alpha` ∈ {1e-4; 1e-2} × `learning_rate_init` ∈ {1e-3; 1e-2} | 8 |

Grade **completa** (para o dado real; 325 combinações no total, 1625 ajustes com 5 dobras):

| Modelo | Grade completa | Comb. |
|---|---|---|
| RL | `C` ∈ {1e-3; 1e-2; 1e-1; 1; 10; 100} | 6 |
| NB | `var_smoothing` ∈ {1e-11; 1e-10; 1e-9; 1e-8; 1e-7; 1e-6; 1e-5} | 7 |
| KNN | `n_neighbors` ∈ {3; 5; 7; 11; 15; 21; 31} × `weights` (2) × `p` ∈ {1; 2} | 28 |
| DT | `criterion` ∈ {gini; entropy} × `max_depth` ∈ {2; 3; 4; 5; 6; 8; 10; None} × `min_samples_leaf` ∈ {1; 3; 5; 10; 20} | 80 |
| RF | `n_estimators` ∈ {100; 300; 500} × `max_depth` ∈ {None; 5; 10; 20} × `min_samples_leaf` ∈ {1; 3; 5} × `max_features` ∈ {sqrt; 0,3} | 72 |
| GB | `n_estimators` ∈ {100; 200; 400} × `learning_rate` ∈ {0,01; 0,05; 0,1} × `max_depth` ∈ {2; 3; 4} × `subsample` ∈ {0,8; 1} | 54 |
| XGBoost | `n_estimators` ∈ {100; 300; 500} × `learning_rate` ∈ {0,01; 0,05; 0,1} × `max_depth` ∈ {3; 5; 7} × `subsample` ∈ {0,8; 1} | 54 |
| MLP | `hidden_layer_sizes` ∈ {(32,); (64,); (64, 32); (128, 64)} × `alpha` ∈ {1e-4; 1e-3; 1e-2} × `learning_rate_init` ∈ {1e-3; 1e-2} | 24 |

Se a grade completa ficar lenta com os dados reais, a alternativa é `RandomizedSearchCV` (sorteia combinações em vez de testar
todas); fica como decisão do passo 2.

## 6. Testes (`tests/test_modelos.py`, cerca de 10)

1. O catálogo tem exatamente os 8 nomes, nas duas grades, e cada entrada é `(estimador, grade)`.
2. **Todo hiperparâmetro de toda grade existe no pipeline:** `pipeline.set_params(**uma_combinacao)` funciona para a primeira
   e a última combinação de cada modelo, nas duas grades (pega erro de digitação e parâmetro descontinuado).
3. Cada um dos 8 pipelines, ajustado numa amostra do treino (cerca de 200 linhas), faz `predict_proba` com formato
   `(n, 2)`, sem `NaN` e com linhas somando 1.
4. `clone(pipeline)` funciona (necessário para o GridSearchCV), inclusive com `GradientBoostingBalanceado`.
5. `desbalanceamento="class_weight"`: RL, DT e RF ficam com `class_weight="balanced"`; XGBoost e GB usam os subtipos
   `*Balanceado`; KNN, NB e MLP não são alterados.
6. `GradientBoostingBalanceado` dá probabilidades diferentes do GB simples em dados desbalanceados (o peso é aplicado).
7. `desbalanceamento` ou `grade` desconhecidos levantam `ValueError`.
8. O primeiro passo de todo pipeline é o pré-processador (`prep`), e ele **ainda não está ajustado** (nada aprendido antes da
   validação cruzada).
9. Mesma semente, mesmo resultado: dois ajustes do mesmo pipeline dão as mesmas probabilidades.

## 7. Verificação ponta a ponta

1. `.venv\Scripts\python.exe -m pytest -q`: os 38 anteriores mais os cerca de 10 novos, todos passando.
2. `.venv\Scripts\python.exe modelos.py --listar`: a tabela do catálogo, com 59 combinações na enxuta e 325 na completa.
3. Conferência manual de um pipeline de cada família ajustado em `data/treino.csv`, só para ver que roda (sem olhar métricas:
   o teste continua lacrado).

## 8. Decisões do usuário (2026-10-29) e pontos em aberto

Decididos para este passo:

1. **P7:** SVM e redes profundas ficam para depois dos 8 modelos rodarem de ponta a ponta.
2. **P6:** criar o subtipo mínimo `GradientBoostingBalanceado`.
3. **Grades (P3):** os valores iniciais da seção 5 foram aprovados. A orientadora é da enfermagem e não revisará os
   hiperparâmetros: são decisão técnica nossa. Se a grade completa ficar lenta com os dados reais, usa-se a enxuta (ou a
   `RandomizedSearchCV`, ver seção 5).

Para decidir só no **passo 2** (não bloqueiam este):

- **Como estimar o desempenho:** `GridSearchCV` simples (a nota do vencedor é um pouco otimista), ou **validação cruzada
  aninhada** (`cross_val_score` em volta do `GridSearchCV`, que combina as duas funções citadas no PDF e dá a nota honesta,
  mas custa 5 vezes mais).
- **De onde vêm as curvas ROC "de todos os modelos"** (p. 47 do PDF): das predições fora da dobra (no treino) ou do teste.
  Abrir o teste para os 8 modelos e depois escolher o melhor é usar o teste para selecionar, então a recomendação é usar as
  predições fora da dobra para a comparação e o teste só para o modelo escolhido.
- **Regra dos 70%** do PDF para SVM e redes profundas.

## 9. Riscos e observações

- **KNN, NB e MLP não aceitam `class_weight`:** no modo `class_weight` eles rodam sem ajuste, então a comparação entre
  famílias não é totalmente "justa" nesse modo. O modo SMOTE (passo 3) equilibra isso, porque age nos dados e não no modelo.
- **O pré-processador decide quais colunas ganham o indicador "foi medido?" olhando o treino inteiro**, não cada dobra.
  Isso é uma decisão estrutural (não usa o alvo nem valores), com efeito desprezível, mas fica registrada.
- **Naive Bayes Gaussiano com colunas 0/1 e one-hot** é uma aproximação grosseira da suposição de normalidade; serve como
  linha de base, como o próprio PDF descreve.
- **Avisos de convergência** (MLP, RL) podem aparecer; não são suprimidos, para não esconder um problema.
- **Paralelismo:** o XGBoost usa vários núcleos por conta própria; combinado ao `n_jobs` do GridSearchCV pode competir por
  CPU. Será tratado no passo 2.
- **O seed não tem valor científico:** este passo só valida o encanamento.
- **Probabilidades:** `class_weight` e SMOTE as deslocam para cima (ver capítulo 6.4 do material); a calibração fica como
  proposta à orientadora e não faz parte deste passo.

## 10. Documentação a atualizar ao fim do passo

Este arquivo (status, o que foi implementado e desvios), o `README.md` (etapa 3 com link para este plano e comando
`--listar` em "Como usar") e o capítulo de modelos do material, se algo mudar. Commit só depois da confirmação.

## 11. Implementação e desvios

**Entregue** em `modelos.py`: `catalogo(grade, desbalanceamento, seed)`, `construir_pipeline(preprocessador,
estimador)`, `GradientBoostingBalanceado`, `n_combinacoes` e o comando `--listar`. Em `tests/test_modelos.py`: 12 testes.
Verificação: **50 testes passam** (38 anteriores + 12 novos); `--listar` confirma 59 combinações na grade enxuta e 325 na
completa (295 e 1625 ajustes com 5 dobras).

**Desvios do plano:**

| Desvio | Motivo |
|---|---|
| `construir_pipeline(preprocessador, estimador)` **não** recebe `desbalanceamento` (o plano da seção 4 previa o parâmetro) | O ajuste de classe por modelo já é feito em `catalogo`. O parâmetro só fará sentido no passo 3, quando o SMOTE for um passo do pipeline |
| `construir_pipeline` **clona** o pré-processador | O `Pipeline` do scikit-learn ajusta os passos no próprio objeto (sem `memory`); sem o clone, os 8 modelos compartilhariam um pré-processador ajustado. Verificado por mutação: sem o `clone`, o teste de isolamento falha |
| 12 testes em vez de cerca de 10 | Acrescentados: o modo `"nenhum"` não pesa nada, e o alvo não está em `X`. O teste de "pré-processador não ajustado" olha o `ColumnTransformer` (`prep["colunas"]`), porque `check_is_fitted` no `Pipeline` inteiro sempre passa: o último passo (`FunctionTransformer`) não tem estado |
| Comentário de `preprocessamento.py` corrigido | Falava de `excluir_vazamento` (passo 4); agora cita `excluir_sinalizadas` na etapa de modelos |

**Atualizações posteriores (passo 2 e revisão adversarial, ver [`modelos-passo2-plan.md`](modelos-passo2-plan.md)):**
`XGBClassifier` passou a `n_jobs=1` (o paralelismo fica na busca em grade); `catalogo` devolve cópias das listas das
grades; a mensagem de `--listar` agora conta os ajustes reais de `validacao.py` (busca simples + aninhada 5x5 = 30 por
combinação: 1770 na enxuta e 9750 na completa, fora os refits); `tests/test_modelos.py` passou a 18 testes (valores reais
das grades ajustados, semente, parâmetros fixos, peso do usuário no GB).

**Atualização de 2026-10-29 (peso do XGBoost por ajuste):** o `scale_pos_weight` calculado com a prevalência do treino inteiro
foi substituído por `XGBClassifierBalanceado`, que calcula o peso dos positivos (negativos/positivos) com o `y` de cada
`fit`, ou seja, de cada dobra. Motivo: a regra do projeto (tudo o que é aprendido vem só do treino de cada rodada) vale
mesmo para um número escalar, e o GB já funcionava assim. Com isso, o argumento `prevalencia` do `catalogo` foi removido
(deixaria de ter uso). A razão de pesos é a mesma do `scale_pos_weight`, mas o resultado não é numericamente idêntico: numa
amostra de teste, a diferença máxima nas probabilidades foi de 0,08, com médias iguais. Testes: `tests/test_modelos.py`
passou a 20 (peso por ajuste verificado com um espião; peso do usuário preservado; falha se não houver positivos); das 3
mutações aplicadas ao novo código, as 3 foram detectadas.

**Observações da implementação:** nenhum modelo foi ajustado nos dados reais do `data/`; o teste lacrado continua intocado. Os
testes ajustam os modelos só em amostras de 300 linhas do treino sintético, para validar o encanamento.
