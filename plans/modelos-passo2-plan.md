# Plano de implementação: passo 2 (busca em grade e validação cruzada) da etapa de modelos

Segundo passo da etapa 3 do README. Depende do catálogo do passo 1 ([`modelos-passo1-plan.md`](modelos-passo1-plan.md)).
Conceitos de apoio: capítulo 4 de `docs/material-de-estudo.pdf` (dobras, GridSearchCV, validação aninhada) e seção 6.4
(calibração, relevante para as probabilidades fora da dobra).

| | |
|---|---|
| **Status** | Decisões do usuário em 2026-10-29; **implementado**, com revisão adversarial (seção 6) |
| **Branch** | `create-pipeline` |
| **Código** | `validacao.py` (novo), `tests/test_validacao.py` (novo); ajustes pequenos em `modelos.py` e `preprocessamento.py` (seção 6) |

## 1. Escopo e decisões do usuário

Para cada um dos 8 modelos, com a grade enxuta, só no conjunto de **treino** (o teste nem é lido):

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P1 | Como estimar o desempenho | **Ambas**: busca simples (`GridSearchCV`, 5 dobras) e validação cruzada **aninhada** (5 dobras externas × busca de 5 dobras interna) | Usuário |
| P2 | De onde vêm as probabilidades para as curvas ROC do passo 4 | **Fora da dobra**, da validação aninhada (cada paciente é previsto por um modelo que não o viu, nem na busca de hiperparâmetros). O teste só será aberto para o modelo escolhido | Usuário |
| P3 | Execução do catálogo | Os 8 modelos rodam por inteiro com a grade enxuta | Usuário |
| P4 | Métrica de escolha | F1 (limiar 0,5), como no PDF. Acurácia, precisão, recall e AUROC são registradas. Empate de F1 é desfeito pela AUROC média (seção 6) | PDF (Quadro 5) e revisão |
| P5 | Dobras | `StratifiedKFold(5, shuffle=True, random_state=seed)`; as dobras externas são as mesmas para os 8 modelos (comparação pareada) | Proposta |
| P6 | Paralelismo | `n_jobs` do `GridSearchCV` (padrão `-1`); o XGBoost fica com 1 thread (`n_jobs=1` em `modelos.py`) | Proposta |
| P7 | `error_score="raise"` | Um ajuste que falha interrompe a execução, em vez de virar NaN calado | Proposta |
| P8 | Saída | Tabela comparativa e as probabilidades fora da dobra, gravadas em `data/resultados/` (fora do git) | Proposta |

Fora do escopo: SMOTE e `excluir_sinalizadas` (passo 3); curvas ROC, avaliação no teste e calibração (passo 4);
`RandomizedSearchCV` e grade completa.

## 2. Desenho

```
comparar_modelos(X, y, ids, grade, desbalanceamento, modelos, ...) -> (tabela, oof)
   para cada modelo:
      avaliar_modelo(pipeline, grade_hiper, X, y):
         busca simples : GridSearchCV no treino inteiro            -> nota otimista do vencedor
         aninhada      : para cada dobra externa
                           GridSearchCV nas 4 dobras de estudo     -> melhor combinação
                           previsão na dobra que ficou de fora     -> métricas da dobra + oof
```

Colunas da tabela: `f1`, `f1_desvio`, `auroc`, `precisao`, `recall`, `acuracia` (médias e desvio das 5 dobras externas),
`f1_busca_simples`, `otimismo_f1` (= busca simples − aninhada), `melhores_params` (da busca simples) e `segundos`.

Linha de comando: `python validacao.py --grade enxuta --desbalanceamento class_weight [--modelos RL DT] [--n-jobs N]`.

## 3. Resultado com o seed (execução completa, grade enxuta, `class_weight`, 800 pacientes, cerca de 3 minutos)

Valores **sem valor científico** (o sinal foi plantado pelo gerador): só mostram que o encanamento funciona.

| Modelo | F1 aninhado | dp(F1) | AUROC | Precisão | Recall | F1 busca simples | Otimismo |
|---|---|---|---|---|---|---|---|
| RL | 0,969 | 0,001 | 0,999 | 0,958 | 0,981 | 0,972 | 0,003 |
| XGBoost | 0,968 | 0,019 | 0,997 | 0,988 | 0,950 | 0,972 | 0,003 |
| GB | 0,962 | 0,014 | 0,998 | 0,975 | 0,950 | 0,964 | 0,002 |
| RF | 0,956 | 0,029 | 0,998 | 0,970 | 0,944 | 0,961 | 0,005 |
| MLP | 0,925 | 0,039 | 0,995 | 0,934 | 0,919 | 0,922 | -0,003 |
| NB | 0,810 | 0,065 | 0,953 | 0,866 | 0,762 | 0,810 | 0,000 |
| DT | 0,756 | 0,051 | 0,839 | 0,725 | 0,794 | 0,782 | 0,026 |
| KNN | 0,749 | 0,097 | 0,918 | 0,977 | 0,613 | 0,766 | 0,017 |

(Segunda execução, depois das correções da seção 6; a primeira, antes delas, deu valores muito próximos, o que é esperado:
a correção do pré-processador só muda a estrutura em exames com ausência perto de 20%.) O KNN tem precisão alta e recall
baixo (0,613): sem ajuste de classe no modo `class_weight`, ele deixa passar casos, como previsto no plano do passo 1.
Otimismo de 0,000 (NB) ou negativo (MLP) não é erro: a busca simples e a aninhada usam a mesma semente de dobras, então,
quando a busca reescolhe a mesma combinação, as notas coincidem; e a nota aninhada pode superar a simples por acaso.

## 4. Testes (`tests/test_validacao.py`, 16)

Formato da tabela e do `oof`; **vazamento** (alvo embaralhado com uma árvore que decora: a AUROC fora da dobra fica perto de
0,5); `oof` **informativo** com o alvo real; métricas **conferidas à mão** num caso pequeno; desempate pela AUROC; linha da
tabela coerente com as dobras e com a busca (média, desvio com ddof=1, otimismo, vencedor = maior F1); argumentos do
catálogo; alinhamento com índice embaralhado; reprodutibilidade; hiperparâmetro inválido para a execução; entradas
inválidas (poucos positivos, rótulos que não são 0 e 1, `X` que não é DataFrame, modelos repetidos); e o CLI de ponta a ponta.

## 5. Observações

- O desvio entre dobras (ddof=1) **não é um intervalo de confiança**: as dobras compartilham a maior parte do treino.
- O `scale_pos_weight` do XGBoost usa a prevalência do treino inteiro (diferença de ordem 1/n entre dobras estratificadas;
  é um número, não um valor aprendido da dobra de validação). Limitação aceita e documentada.
- O F1 em limiar 0,5 é uma métrica ruidosa para KNN (probabilidades discretas), e os modelos sem ajuste de classe (KNN, NB,
  MLP) não competem em igualdade com os ponderados no modo `class_weight`. O SMOTE (passo 3) ajuda a equilibrar.
- **Para o passo 3 (`excluir_sinalizadas`):** `tratar_ausentes_estruturais` exige a coluna `hemocultura_resultado`; tirar as
  colunas sinalizadas de `X` antes do pré-processador dá `KeyError`. O interruptor precisará de ajuste no pré-processador
  (por exemplo, tolerar a ausência dessas colunas). Além disso, no seed a RL chega a AUROC 0,999 mesmo sem os totais "dias":
  há preditores quase separáveis plantados pelo gerador, então a análise de sensibilidade com o seed não dirá muito.

## 6. Revisão adversarial (antes do commit) e correções

Duas revisões independentes (uma de vazamento e método, outra de engenharia e qualidade dos testes), sem acesso ao conjunto
de teste e sem alterar arquivos. Cada achado foi confirmado antes de corrigir.

| Achado | Gravidade | Correção |
|---|---|---|
| A estrutura do pré-processador (quais exames têm indicador "foi medido?"; quais colunas são binárias) era decidida com o treino inteiro, **inclusive a dobra de validação**: mudava o modelo em 4 das 5 dobras | Média | Decisão movida para o `fit` (seletores de coluna no `ColumnTransformer`). Teste novo; falha com o código antigo |
| O filtro de avisos por `PYTHONWARNINGS` não funcionava (o interpretador já havia iniciado) | Média | `warnings.filterwarnings` no processo principal |
| Poucos positivos não falhavam (F1 = 0 e AUROC ~ 0,5 viravam "resultado") | Média | `ValueError` se uma classe tiver menos de `n_dobras²` exemplos |
| `y` com rótulos {1, 2} era aceito, tratando a classe negativa como positiva; `y` em array quebrava | Baixa | `_preparar` normaliza e valida (0/1, `DataFrame`, tamanhos) |
| Empate de F1 (todo F1 = 0) escolhia o primeiro da grade em silêncio | Baixa | `refit` próprio: desempata pela AUROC média |
| `--modelos RL RL` rodava duas vezes e sobrescrevia; `modelos="NB"` virava `["N", "B"]`; grades devolvidas compartilhavam listas | Baixa | Normalização e `list(valores)` |
| Testes fracos: 31 mutações plausíveis aplicadas, várias sobreviviam (`oof` invertido, precisão↔recall, `best_index_`, desvio, otimismo, argumentos do catálogo, CLI...) | Média/alta | 9 testes novos em `test_validacao.py` e 6 em `test_modelos.py`; das 21 mutações reaplicadas, **21 morrem** |
| Documentação: README sem `validacao.py` e sem plano do passo 2; contagem de ajustes subestimada em `--listar` | Baixa | Este plano, README e mensagem do `--listar` corrigidos |

Registrados sem mudar o código: `scale_pos_weight` com o `y` inteiro e o desvio não ser intervalo de confiança (seção 5).
Não confirmado pelas revisões e mantido: nenhum vazamento do alvo (alvo embaralhado nos 8 modelos: AUROC entre 0,49 e 0,58;
resultado idêntico com 1 ou vários núcleos).

## 7. Verificação (2026-10-29)

1. `pytest -q`: **73 testes passam** (17 do gerador, 22 do pré-processador, 18 de modelos, 16 de validação), cerca de 2,5 min.
2. `python validacao.py --grade enxuta --desbalanceamento class_weight`: os 8 modelos rodam por inteiro (cerca de 3 min com
   16 núcleos), **0 avisos** no log, arquivos gravados em `data/resultados/`.
3. Mutações: das 21 reaplicadas em uma cópia do projeto, 21 são detectadas pela suíte; a cópia sem mutação passa.
4. `data/teste.csv` não é lido por nenhum código deste passo.
