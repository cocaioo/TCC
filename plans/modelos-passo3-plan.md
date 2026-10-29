# Plano de implementação: passo 3 (SMOTE e `excluir_sinalizadas`) da etapa de modelos

Terceiro passo da etapa 3 do README. Depende do catálogo e da validação dos passos anteriores
([passo 1](modelos-passo1-plan.md), [passo 2](modelos-passo2-plan.md)) e da redução do passo 4 do pré-processamento
([seção 0](preprocessamento-passo4-plan.md)). Conceitos de apoio: capítulo 5 (desbalanceamento, SMOTE) e seção 6.4
(calibração) de `docs/material-de-estudo.pdf`.

| | |
|---|---|
| **Status** | Plano escrito em 2026-10-29; **aguardando aprovação** (nada implementado) |
| **Branch** | `create-pipeline` |
| **Código previsto** | `modelos.py`, `preprocessamento.py`, `validacao.py`, `tests/test_modelos.py`, `tests/test_preprocessamento.py`, `tests/test_validacao.py` (sem arquivos novos) |

## 1. Contexto e escopo

Dois recursos que o PDF e a conversa pedem e que ainda não existem:

1. **SMOTE** como terceiro modo de lidar com o desbalanceamento (`"nenhum"`, `"class_weight"`, `"smote"`). O PDF prevê avaliar
   "reamostragem sintética e ajuste de pesos de classe" (p. 47). O SMOTE age **nos dados**, então também cobre KNN, NB e MLP, que
   não têm `class_weight` no scikit-learn.
2. **`excluir_sinalizadas`**: treinar sem a "zona cinzenta" (`dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias`,
   `hemocultura_resultado`, `hemocultura_agente`, `perfil_sensibilidade`), como análise de sensibilidade. Não "corrige" vazamento:
   mostra quanto o desempenho depende de variáveis que podem ser consequência da sepse.

Fora do escopo: avaliação no teste, curvas ROC e calibração (passo 4); SMOTE-NC e outras variantes (ver P2); ajuste de
`sampling_strategy` ou `k_neighbors` como hiperparâmetros; KNN como imputador (adiado).

## 2. Medições preliminares (seed, treino de 800 pacientes × 132 colunas)

Experimentos feitos antes de escrever o plano, para o desenho não depender de suposição:

| Fato | Medida | Consequência |
|---|---|---|
| O `Pipeline` do `imbalanced-learn` **recusa pipelines aninhados** (`TypeError: All intermediate steps of the chain should not be Pipelines`), e o pré-processador é um `Pipeline` do scikit-learn | Erro reproduzido | Os passos do pré-processador precisam ser **achatados** dentro do pipeline (P3) |
| Pipeline achatado (`estrutural`, `log1p`, `colunas`, `float`, `smote`, `modelo`) | Ajusta e prevê normalmente; `get_params` mantém `modelo__C` etc. | As grades dos passos 1 e 2 **não mudam** |
| `SMOTE` aceita o `DataFrame` que o pré-processador devolve | 640/160 → 640/640 | Funciona sem adaptação |
| Colunas 0/1 originais "sujas" pelo SMOTE | **65 de 65** colunas passam a ter valores fracionários, em cerca de **9,5%** das linhas de cada uma | É o problema do "sexo = 0,3" (capítulo 5.3 do material). Limitação aceita nesta versão (P2) |
| Protótipo de SMOTE-NC com máscara de colunas 0/1 calculada no `fit` | **Falhou** (`ValueError`), sem investigação | Não é prova de inviabilidade; fica fora do escopo (P2) |
| RL, 5 dobras, seed: AUROC fora da dobra sem reamostragem × com SMOTE | 0,999 × 0,999; probabilidade média prevista 0,196 × 0,202 (prevalência real 0,200) | No seed, o SMOTE **não** inflou as probabilidades: o sinal plantado é forte. Em dado real com sinal fraco, o efeito descrito no capítulo 6.4 deve aparecer |
| Sem as 7 colunas sinalizadas | 132 → **110** colunas (4 numéricas + 18 colunas one-hot das 3 categóricas), valor medido no plano do passo 4 | Referência para o teste do interruptor (a reconferir após a refatoração do passo 2) |

## 3. Decisões

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P1 | Novo modo | `DESBALANCEAMENTO = ("nenhum", "class_weight", "smote")`. Em `"smote"`, `catalogo` devolve os mesmos estimadores de `"nenhum"` (sem `class_weight`, sem `*Balanceado`), para não corrigir o desbalanceamento duas vezes | Proposta |
| P2 | Tipo de SMOTE | `SMOTE` simples (`k_neighbors=5`, `sampling_strategy="auto"`: iguala as classes, `random_state=seed`). SMOTE-NC e variantes ficam para depois, se a limitação das colunas fracionárias (seção 2) importar | Proposta; **precisa da sua confirmação** |
| P3 | Montagem do pipeline | `construir_pipeline(preprocessador, estimador, desbalanceamento="nenhum", seed=42)` devolve **sempre** um `Pipeline` do `imbalanced-learn` com os passos do pré-processador achatados: `[*clone(prep).steps, ("smote", SMOTE(...)), ("modelo", estimador)]`; o passo `smote` só existe no modo `"smote"`. O `Pipeline` do `imblearn` se comporta como o do scikit-learn quando não há amostrador | Proposta (limite do `imblearn`); **precisa da sua confirmação** |
| P4 | Consequência de P3 | O passo `"prep"` deixa de existir: os testes que usam `pipe["prep"]` passam a usar os passos achatados (`pipe["colunas"]`, `pipe.steps[0][0] == "estrutural"`). Nada mais muda (grades, `clone`, `GridSearchCV`) | Consequência |
| P5 | O SMOTE só roda no treino de cada rodada | Garantido pelo `imblearn.Pipeline` (reamostra no `fit`, e nunca no `predict`), dentro da busca interna e das dobras externas. Testado (seção 5), inclusive com um controle positivo que mostra o vazamento de reamostrar antes de dividir | Plano geral (conceito C1/capítulo 5) |
| P6 | `excluir_sinalizadas` | Parâmetro de `construir_preprocessador(X_treino, limiar_indicador, excluir_sinalizadas=False)`. **As colunas continuam em `X`**: são omitidas dos ramos do `ColumnTransformer` (e ele as descarta, `remainder="drop"`) | Plano do passo 4 (P20), ajustado |
| P7 | Por que não remover as colunas de `X` | `tratar_ausentes_estruturais` exige `hemocultura_resultado`; tirar as colunas antes do pré-processador dá `KeyError` (achado da revisão do passo 2). Omiti-las dos ramos evita isso | Revisão adversarial |
| P8 | O que **não** sai | As flags `cvc`, `vm_invasiva` e `npt` (exposição sim/não) permanecem, como no plano do passo 4 | Plano do passo 4 |
| P9 | Integração com a validação | `comparar_modelos(..., excluir_sinalizadas=False)`; na linha de comando, `--excluir-sinalizadas`; os arquivos de saída ganham o sufixo `_sem_sinalizadas` | Proposta |
| P10 | Ajuda para comparar | Função `sensibilidade(tabela_com, tabela_sem)` (F1, AUROC e a queda por modelo) e a opção `--comparar-sinalizadas`, que roda os dois cenários e imprime a diferença | Proposta, **opcional** (ver seção 7) |

## 4. Código a escrever

- `modelos.py`: `"smote"` em `DESBALANCEAMENTO`; `catalogo` trata `"smote"` como `"nenhum"` nos estimadores;
  `construir_pipeline` com o `imblearn.Pipeline` achatado e o passo `smote` (P3); `imprimir_catalogo` mostra a linha do SMOTE.
- `preprocessamento.py`: `excluir_sinalizadas` em `construir_preprocessador`; os seletores (`_selecionar_binarias`,
  `_selecionar_numericas`, `_selecionar_com_indicador`) e a lista de categóricas passam a ignorar `COLS_SINALIZADAS` quando o
  interruptor está ligado. Sem outra mudança de comportamento com o interruptor desligado (132 colunas no seed).
- `validacao.py`: `excluir_sinalizadas` e `desbalanceamento` repassados a `construir_preprocessador` e `construir_pipeline`
  (com `seed`); `--excluir-sinalizadas` e o sufixo de arquivo; `sensibilidade` e `--comparar-sinalizadas` se P10 for aprovado.

## 5. Testes (cerca de 12 novos)

**SMOTE**
1. `catalogo(desbalanceamento="smote")` aceita o modo e devolve estimadores iguais aos de `"nenhum"` (sem pesos).
2. O passo `smote` está entre `float` e `modelo` no modo `"smote"` e **não existe** nos outros dois.
3. Os 8 pipelines ajustam e dão `predict_proba` válido no modo `"smote"`.
4. **Só no treino:** com um estimador espião, o `fit` recebe as classes iguais (negativos = positivos) e o `predict_proba`
   recebe exatamente as linhas pedidas, sem reamostrar.
5. **Sem vazamento:** com o alvo embaralhado e uma árvore que decora, a AUROC fora da dobra com SMOTE dentro do pipeline fica
   perto de 0,5; **controle positivo:** reamostrar *antes* de dividir dá AUROC claramente maior (o teste prova que detectaria o
   vazamento). Os limites exatos serão calibrados na implementação.
6. Mesma semente, mesmo resultado; sementes diferentes, resultados diferentes.
7. `comparar_modelos(desbalanceamento="smote")` roda e o `oof` continua alinhado.

**`excluir_sinalizadas`**
8. Com o interruptor ligado, a matriz tem **110** colunas (valor de referência, a reconferir), e não restam as 4 numéricas nem as
   colunas one-hot das 3 categóricas; as flags `cvc`, `vm_invasiva` e `npt` permanecem.
9. As colunas restantes são **idênticas** (mesmos valores) às da versão com o interruptor desligado.
10. Funciona com `X` completo (sem `KeyError`): a regressão do achado da revisão.
11. A decisão continua sendo tomada no `fit` (o teste do passo 2 sobre a estrutura segue passando com o interruptor ligado).
12. `comparar_modelos(excluir_sinalizadas=True)` repassa o interruptor; o CLI grava os arquivos com o sufixo; `sensibilidade`
    calcula as diferenças (se P10).

Os testes que usam `pipe["prep"]` são ajustados (P4); a suíte atual (75) deve continuar passando.

## 6. Verificação ponta a ponta

1. `.venv\Scripts\python.exe -m pytest -q`: os 75 anteriores (com os ajustes de P4) mais os cerca de 12 novos, todos passando.
2. `.venv\Scripts\python.exe preprocessamento.py` sem opções: continua `800 x 132` (regressão).
3. `.venv\Scripts\python.exe validacao.py --desbalanceamento smote`: os 8 modelos rodam por inteiro, sem avisos.
4. `.venv\Scripts\python.exe validacao.py --excluir-sinalizadas`: os 8 rodam por inteiro, matriz com 110 colunas.
5. Mutações aplicadas a uma cópia do projeto (SMOTE aplicado antes de dividir, `smote` no predict, interruptor ignorado, coluna
   sinalizada que escapa), todas devem ser detectadas.
6. Revisão adversarial antes do commit, como nos passos anteriores.

## 7. Pontos que dependem de você

1. **P2:** `SMOTE` simples agora, e SMOTE-NC só se as colunas fracionárias se mostrarem um problema (recomendado), ou tentar o
   SMOTE-NC já neste passo (o protótipo rápido falhou e exigiria investigação).
2. **P3:** achatar sempre os passos do pré-processador (recomendado: um único formato, mas muda 3 testes) ou só no modo `"smote"`
   (menos mudança agora, mas o pipeline teria formas diferentes conforme o modo).
3. **P10:** incluir a função de comparação e `--comparar-sinalizadas` (poupa conta manual, cerca de 20 linhas) ou deixar só o
   interruptor e comparar os dois CSVs à mão.

## 8. Riscos e observações

- **Colunas fracionárias:** o SMOTE cria valores como 0,3 em colunas 0/1 (65 de 65 colunas, cerca de 9,5% das linhas de cada
  uma). Árvores e boosting lidam bem; KNN e MLP podem ser mais sensíveis. Fica registrado, não corrigido.
- **MLP com `early_stopping`:** a validação interna que ele separa sairia dos dados **já reamostrados**, então pontos sintéticos
  vizinhos dos reais entram nessa validação e a parada precoce fica otimista. Efeito esperado pequeno; observar.
- **SMOTE e poucos positivos:** `k_neighbors=5` exige pelo menos 6 positivos em cada treino de rodada. A validação do passo 2
  (mínimo de 25 por classe) garante cerca de 16 nos treinos da busca interna.
- **Probabilidades:** o SMOTE tende a elevar as probabilidades em dado de sinal fraco (capítulo 6.4). No seed não apareceu; a
  calibração segue como proposta à orientadora, fora deste passo.
- **Custo:** o SMOTE acrescenta pouco tempo por ajuste, mas a busca aninhada multiplica; a execução completa deve ficar na casa
  de poucos minutos a mais.
- **O seed não tem valor científico:** a análise de sensibilidade com ele diz pouco (a RL chega a AUROC 0,999 mesmo sem os totais
  "dias": há preditores quase separáveis plantados pelo gerador). O que se valida é o encanamento.
- **A "zona cinzenta" só se resolve com dado real:** a recomendação de contar `dias_utin`, `vm_dias`, `cvc_dias` e `npt_dias` até o
  momento da suspeita, e não até a última evolução, depende do conector com o prontuário.

## 9. Documentação a atualizar ao fim do passo

Este arquivo (status, o que foi implementado, desvios e resultados), o `README.md` (passo 3 como feito e os comandos novos) e,
se algo mudar, o capítulo de desbalanceamento do material de estudo. Commit só depois da confirmação.
