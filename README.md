# Predição de sepse neonatal tardia com aprendizado de máquina

Pipeline de aprendizado de máquina para estimar a probabilidade de **sepse neonatal tardia** (CID-10 A41.9 / P36) em
recém-nascidos internados em Unidade de Terapia Intensiva Neonatal (UTIN), a partir de dados do prontuário eletrônico
(SOUL MV). O desenho do estudo está em `Projeto_sepse_07_07.pdf` e segue o processo KDD (Descoberta de Conhecimento em Bases
de Dados).

O problema é de classificação binária supervisionada (desfecho `sepse_tardia`: 0 ou 1), com saída em probabilidade. Os dados
reais dependem de aprovação em comitê de ética e do conector com o prontuário; até lá, o pipeline é desenvolvido e testado com
**dados sintéticos** gerados pelo próprio repositório. As métricas obtidas com eles **não têm valor científico**: servem para
validar o funcionamento do pipeline e deixá-lo configurado para os dados reais.

## Etapas

| Etapa | Descrição | Módulo | Estado |
|---|---|---|---|
| Dados sintéticos | Geração de um CSV com 1 linha por recém-nascido (99 colunas), prevalência configurável, sinal clínico plausível, exames seriados e valores ausentes | `gerador_seed.py` | Concluída |
| Pré-processamento e transformação | Auditoria dos dados; divisão em treino (80%) e teste (20%), estratificada por desfecho; pré-processador (hemocultura, `log1p` nos exames de cauda longa, imputação pela mediana, indicador de ausência, padronização, one-hot) | `preprocessamento.py` | Concluída |
| Modelos (mineração de dados) | Catálogo dos 8 algoritmos do Quadro 4 do projeto (RL, NB, KNN, DT, RF, GB, XGBoost, MLP) com grades de hiperparâmetros; busca em grade e validação cruzada (k = 5) simples e aninhada; tratamento do desbalanceamento por pesos de classe | `modelos.py`, `validacao.py` | Em andamento |
| Avaliação | Avaliação final no conjunto de teste e curvas ROC; calibração das probabilidades (proposta) | | Prevista |
| Interpretação | Valores SHAP do melhor modelo | | Prevista |

### Passos e planos

Cada etapa é dividida em passos numerados, com um plano versionado em `plans/<etapa>-passoN-plan.md` (decisões, implementação e
desvios). A numeração recomeça a cada etapa, e por isso os passos são citados sempre com o nome da etapa.

| Etapa | Passo | Conteúdo | Estado |
|---|---|---|---|
| Pré-processamento | 1 | Carregamento e auditoria | Concluído |
| Pré-processamento | 2 | Divisão treino/teste | Concluído |
| Pré-processamento | 3 | Pré-processador | Concluído |
| Pré-processamento | 4 | Estratégias de imputação e análise sem as variáveis sinalizadas | Reduzido: apenas a análise sem as variáveis sinalizadas permanece, incorporada a Modelos, passo 3 |
| Pré-processamento | 5 | Fechamento da etapa | Concluído |
| Modelos | 1 | Catálogo de modelos e montagem dos pipelines | Concluído |
| Modelos | 2 | Busca em grade e validação cruzada, simples e aninhada | Concluído |
| Modelos | 3 | SMOTE e interruptor `excluir_sinalizadas` | Plano elaborado |
| Modelos | 4 | Avaliação no teste e curvas ROC | Previsto |

## Princípios metodológicos

- O conjunto de teste permanece lacrado até a avaliação final.
- Tudo o que é aprendido dos dados (medianas, escalas, estrutura do pré-processador, pesos de classe) provém apenas do treino de
  cada rodada; a dobra de validação não influencia o ajuste. Isso é coberto por testes que falham se o vazamento reaparecer.
- A coluna `dias_ate_dx_sepse` é removida das variáveis de entrada: só existe para quem teve sepse e prevê o desfecho por
  construção (vazamento de alvo).
- As variáveis `dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias` e `hemocultura_*` ("zona cinzenta") permanecem como preditoras, mas
  podem ser consequência da sepse. Com dados reais, recomenda-se contar esses dias até o momento da suspeita.
- O desempenho é estimado de duas formas: busca simples (nota otimista do vencedor) e validação cruzada aninhada (nota honesta).
  As probabilidades "fora da dobra" da validação aninhada são a base das curvas ROC.
- O F1-score (limiar 0,5) é a métrica principal de seleção, conforme o projeto; acurácia, precisão, recall e AUROC também são
  registradas.
- Nos modelos sem suporte a `class_weight` no scikit-learn (KNN, NB e MLP), o modo de pesos de classe não aplica ajuste; o SMOTE
  (Modelos, passo 3) cobre esses casos.

## Como executar

Requer Python 3.11 ou superior. Os comandos são para o Windows (PowerShell), a partir da pasta do projeto.

```powershell
# Instalar as dependências (uma vez)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Gerar os dados sintéticos -> data/seed_sepse.csv
.venv\Scripts\python.exe gerador_seed.py --n 1000 --prevalencia 0.20 --sinal 1.0 --ausentes 1.0 --seed 42

# 2. Auditar e dividir em treino e teste (-> data/treino.csv e data/teste.csv)
.venv\Scripts\python.exe preprocessamento.py --entrada data/seed_sepse.csv --seed 42

# 3. Listar o catálogo de modelos (número de combinações das grades e ajuste de desbalanceamento)
.venv\Scripts\python.exe modelos.py --listar

# 4. Busca em grade e validação cruzada dos 8 modelos, apenas com o treino (cerca de 3 min na grade enxuta)
#    -> data/resultados/comparacao_enxuta_class_weight.csv e oof_enxuta_class_weight.csv
.venv\Scripts\python.exe validacao.py --grade enxuta --desbalanceamento class_weight
#    opções: --modelos RL DT (subconjunto), --n-jobs N, --desbalanceamento nenhum, --grade completa

# Testes automáticos (75; cerca de 2,5 min)
.venv\Scripts\python.exe -m pytest
```

Parâmetros do gerador: `--sinal` (0 = classes indistinguíveis, 1 = efeito completo), `--ausentes` (multiplicador das taxas de
dados ausentes) e `--seed` (a mesma semente gera o mesmo CSV).

## Estrutura

```
gerador_seed.py            gerador de dados sintéticos
preprocessamento.py        auditoria, divisão treino/teste e pré-processador
modelos.py                 catálogo dos 8 algoritmos, grades de hiperparâmetros e montagem dos pipelines
validacao.py               busca em grade e validação cruzada (simples e aninhada)
tests/                     testes automáticos (pytest), um arquivo por módulo
plans/                     planos por passo
docs/                      material de estudo (PDF e fonte LaTeX) com os conceitos do pipeline
data/                      dados e resultados gerados (fora do git)
Projeto_sepse_07_07.pdf    projeto de pesquisa
requirements.txt           dependências
```

Os dados nunca são versionados: `data/` está no `.gitignore` (os dados reais serão privados e os sintéticos são regeneráveis).

## Limitações e pendências

- Os efeitos do gerador sintético são estimativas clínicas, e não derivam de dados reais; os modelos recuperam o que foi
  plantado, e o desempenho com dados reais tende a ser inferior.
- Com cerca de 700 pacientes, 15 a 25% de sepse (105 a 175 casos) e 97 preditoras, há risco de sobreajuste; o projeto não prevê
  seleção de variáveis.
- O desvio entre as dobras da validação não é um intervalo de confiança, pois as dobras compartilham a maior parte do treino.
- Pontos a definir com a orientação do estudo: procedimento de imputação (não detalhado no projeto), calibração das
  probabilidades (ausente do Quadro 5), limiar de decisão diferente de 0,5, seleção de variáveis e a contagem dos dias da zona
  cinzenta até a suspeita.
- SVM e redes neurais profundas, citados no Quadro 4 do projeto, serão avaliados após os 8 modelos do catálogo.
