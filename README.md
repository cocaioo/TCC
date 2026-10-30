# Predição de sepse neonatal tardia com aprendizado de máquina

Pipeline de aprendizado de máquina para estimar a probabilidade de **sepse neonatal tardia** (CID-10 A41.9 / P36) em
recém-nascidos internados em Unidade de Terapia Intensiva Neonatal (UTIN), a partir de dados do prontuário eletrônico
(SOUL MV). O desenho do estudo está em `Projeto_sepse_07_07.pdf` e segue o processo KDD (Descoberta de Conhecimento em Bases
de Dados).

O objetivo do estudo é **desenvolver e avaliar** algoritmos de classificação para esse desfecho. O pipeline testa os 8
algoritmos do Quadro 4 do projeto de ponta a ponta: da tabela de entrada (uma linha por recém-nascido, desfecho
`sepse_tardia` 0 ou 1) até a comparação dos modelos, a avaliação final e a explicação do melhor modelo.

Os dados reais dependem de aprovação em comitê de ética, do conector com o prontuário e de definições de desenho ainda em
aberto (ver `docs/entendimento-do-projeto.md`). Até lá, o pipeline roda com **dados sintéticos** gerados pelo próprio
repositório. As métricas obtidas com eles **não têm valor científico**: servem para verificar o funcionamento do pipeline.

## Estações do pipeline

| # | Estação | Módulo | Estado |
|---|---|---|---|
| 1 | Dados de entrada: CSV sintético com 1 linha por recém-nascido (99 colunas), prevalência configurável, exames seriados e valores ausentes | `gerador_seed.py` | Pronta (sintéticos) |
| 2 | Pré-processamento: auditoria; hemocultura, `log1p` nos exames de cauda longa, imputação pela mediana, indicador de ausência, padronização, one-hot | `preprocessamento.py` | Pronta |
| 3 | Divisão em treino (80%) e teste (20%), estratificada pelo desfecho | `preprocessamento.py` | Pronta |
| 4 | Treino e comparação: 8 algoritmos (RL, NB, KNN, DT, RF, GB, XGBoost, MLP), busca em grade, validação cruzada (k = 5) simples e aninhada; desbalanceamento sem ajuste, com pesos de classe ou com SMOTE | `modelos.py`, `validacao.py` | Pronta |
| 5 | Avaliação final: escolha do vencedor pela validação, teste aberto uma vez, tabela com as métricas do Quadro 5 e a especificidade, intervalo de 95% por bootstrap, curvas ROC e matriz de confusão | `avaliacao.py` | Pronta |
| 6 | Explicação do vencedor com valores SHAP e modelo final salvo (pipeline completo e descrição em JSON) | `explicacao.py` | Pronta |

O plano que conduziu a implementação está em `plans/plano-de-implementacao.md`, e o relatório do que foi feito, com as
justificativas, em `docs/relatorio-do-pipeline.md`.

## O que depende dos dados reais

As estações 3 a 6 não dependem do conteúdo dos dados: recebem uma tabela com uma coluna de identificação, o desfecho 0/1 e
as variáveis preditoras. O que depende dos dados está concentrado em dois lugares:

- **`gerador_seed.py`**: será substituído pela extração real.
- **Constantes do topo de `preprocessamento.py`**: nomes das colunas categóricas, dos exames (sufixos das janelas
  `_pre` e `_pos24h`), dos exames que recebem `log1p`, da coluna de vazamento estrutural e das colunas sinalizadas, além
  dos critérios de inclusão auditados. Quando os dados reais chegarem, o ajuste se concentra nessas listas.

Essa separação é intencional: definições como quem entra na base e qual a referência temporal dos exames ainda serão
discutidas com a orientação, e o pipeline foi preparado para absorvê-las com mudanças pequenas.

## Princípios metodológicos

- O conjunto de teste permanece lacrado até a avaliação final e é aberto uma única vez.
- Tudo o que é aprendido dos dados (medianas, escalas, estrutura do pré-processador, pesos de classe) provém apenas do treino
  de cada rodada; a dobra de validação não influencia o ajuste. Isso é coberto por testes que falham se o vazamento
  reaparecer.
- A coluna `dias_ate_dx_sepse` é removida das variáveis de entrada: só existe para quem teve sepse e prevê o desfecho por
  construção (vazamento de alvo).
- As variáveis `dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias` e `hemocultura_*` ("zona cinzenta") permanecem como preditoras,
  mas podem ser consequência da sepse; o tratamento delas depende das definições em aberto.
- O desempenho é estimado de duas formas: busca simples (nota otimista do vencedor) e validação cruzada aninhada (nota
  honesta). As probabilidades "fora da dobra" da validação aninhada alimentam as curvas ROC do treino.
- O F1-score (limiar 0,5) é a métrica principal de seleção, conforme o projeto; acurácia, precisão, recall, especificidade e
  AUROC também são registradas.
- O vencedor (algoritmo e modo de desbalanceamento) é escolhido pela validação aninhada, antes de o teste ser aberto; o teste
  só mede.
- Nos modelos sem suporte a `class_weight` no scikit-learn (KNN, NB e MLP), o modo de pesos de classe não aplica ajuste; o
  SMOTE, que age nos dados, cobre os 8 modelos e roda só no treino de cada ajuste.

## Como executar

Requer Python 3.11 ou superior. Os comandos são para o Windows (PowerShell), a partir da pasta do projeto.

```powershell
# Instalar as dependências (uma vez)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Gerar os dados sintéticos -> data/seed_sepse.csv
.venv\Scripts\python.exe gerador_seed.py --n 1000 --prevalencia 0.20 --sinal 1.0 --ausentes 1.0 --seed 42

# 2 e 3. Auditar e dividir em treino e teste (-> data/treino.csv e data/teste.csv)
.venv\Scripts\python.exe preprocessamento.py --entrada data/seed_sepse.csv --seed 42

# Listar o catálogo de modelos (número de combinações das grades e ajuste de desbalanceamento)
.venv\Scripts\python.exe modelos.py --listar

# 4. Busca em grade e validação cruzada dos 8 modelos, apenas com o treino, um modo de desbalanceamento por vez
#    (cerca de 3 a 7 min cada na grade enxuta)
#    -> data/resultados/comparacao_enxuta_<modo>.csv, oof_enxuta_<modo>.csv e modelos_enxuta_<modo>/
.venv\Scripts\python.exe validacao.py --grade enxuta --desbalanceamento nenhum
.venv\Scripts\python.exe validacao.py --grade enxuta --desbalanceamento class_weight
.venv\Scripts\python.exe validacao.py --grade enxuta --desbalanceamento smote
#    opções: --modelos RL DT (subconjunto), --n-jobs N, --grade completa

# 5. Avaliação final: escolhe o vencedor pela validação e só então abre o teste
#    -> avaliacao_teste_enxuta.csv, vencedor_enxuta.json, roc_oof_*.png, roc_teste_*.png, confusao_vencedor_enxuta.png
.venv\Scripts\python.exe avaliacao.py --grade enxuta

# 6. SHAP do vencedor e modelo final salvo (cerca de 1 min)
#    -> importancia_shap_enxuta.csv, shap_resumo_enxuta.png, shap_caso_enxuta.png, modelo_final.joblib e .json
.venv\Scripts\python.exe explicacao.py --grade enxuta

# Testes automáticos
.venv\Scripts\python.exe -m pytest
```

Parâmetros do gerador: `--sinal` (0 = classes indistinguíveis, 1 = efeito completo), `--ausentes` (multiplicador das taxas de
dados ausentes) e `--seed` (a mesma semente gera o mesmo CSV).

## Estrutura

```
gerador_seed.py            gerador de dados sintéticos
preprocessamento.py        auditoria, divisão treino/teste e pré-processador
modelos.py                 catálogo dos 8 algoritmos, grades de hiperparâmetros e montagem dos pipelines
validacao.py               busca em grade e validação cruzada (simples e aninhada); salva os modelos finais
avaliacao.py               avaliação final no teste, curvas ROC e matriz de confusão
explicacao.py              valores SHAP do vencedor, modelo final salvo e função prever
tests/                     testes automáticos (pytest), um arquivo por módulo
plans/                     plano de implementação
docs/                      entendimento do projeto, relatório do pipeline (com figuras) e material de estudo
data/                      dados e resultados gerados (fora do git)
Projeto_sepse_07_07.pdf    projeto de pesquisa
requirements.txt           dependências
```

Os dados nunca são versionados: `data/` está no `.gitignore` (os dados reais serão privados e os sintéticos são regeneráveis).

## Limitações e pendências

- Os efeitos do gerador sintético são estimativas clínicas, e não derivam de dados reais; os modelos recuperam o que foi
  plantado, e o desempenho com dados reais tende a ser inferior.
- Com cerca de 700 pacientes, 15 a 25% de sepse (105 a 175 casos) e 97 preditoras, há risco de sobreajuste; o projeto não
  prevê seleção de variáveis.
- O desvio entre as dobras da validação não é um intervalo de confiança, pois as dobras compartilham a maior parte do treino.
- Pontos a definir com a orientação: estão registrados em `docs/entendimento-do-projeto.md`.
