# TCC: predição de sepse neonatal tardia com aprendizado de máquina

Pipeline de aprendizado de máquina para prever **sepse neonatal tardia** (CID-10 A41.9 / P36) em recém-nascidos
internados na UTIN, a partir de dados do prontuário eletrônico (SOUL MV). O desenho do estudo está em
`Projeto_sepse_07_07.pdf` e segue as etapas do processo KDD (Descoberta de Conhecimento em Bases de Dados).

Como os dados reais dependem de aprovação em comitê de ética (CEP) e do conector com o prontuário, o pipeline está
sendo construído e testado com **dados sintéticos** gerados por este repositório. As métricas obtidas com eles
**não têm valor científico**: servem para validar que o pipeline funciona de ponta a ponta.

## Como estamos trabalhando

- **Uma etapa por vez**, com o conceito explicado antes do código.
- **Módulos Python (`.py`)**, sem notebooks.
- **Cada etapa tem um plano versionado em `plans/`**, com decisões, o que foi implementado e histórico, para dar
  rastreabilidade.
- Dados nunca vão para o git: `data/` está no `.gitignore` (os reais serão privados; os sintéticos são regeneráveis).

## Passo a passo

### Etapa 0: dados sintéticos (seed)

| | |
|---|---|
| Estado | **Concluída** |
| Código | `gerador_seed.py`, `tests/test_gerador_seed.py` |
| Plano | [`plans/seed-plan.md`](plans/seed-plan.md) |

Gera um CSV com 1 linha por recém-nascido e 99 colunas (97 preditoras, `id_paciente` e `sepse_tardia`), com as variáveis
dos Quadros 2 e 3 do projeto, desfecho com prevalência configurável (padrão 20%), sinal clínico plausível, exames
seriados (`_pre` e `_pos24h`) e valores ausentes.

### Etapa 1: pré-processamento e transformação (KDD, etapas 2 e 3)

| | |
|---|---|
| Código | `preprocessamento.py`, `tests/test_preprocessamento.py` |
| Plano | [`plans/preprocessamento-plan.md`](plans/preprocessamento-plan.md) |

| Passo | O que é, em linguagem simples | Estado |
|---|---|---|
| 1. Carregar e auditar | Ler o CSV e fazer um "check-up": quantas linhas, quantos ausentes, se há coluna suspeita de vazamento | **Feito** |
| 2. Dividir | Separar 80% dos pacientes para treinar e 20% para o teste final, mantendo a mesma proporção de sepse | **Feito** |
| 3. Pré-processador | Preencher os ausentes, colocar os números na mesma escala e transformar texto em número, aprendendo só do treino | **Feito** |
| 4. Estratégias e interruptores | Opções extras: outras formas de preencher ausentes (KNN e iterativo) e a análise com e sem as variáveis da "zona cinzenta" | Pendente |
| 5. Fechamento | Testes automáticos, atualização do plano e commit | Pendente |

### Etapas seguintes (ainda sem plano)

| Etapa | O que faz |
|---|---|
| 2. Desbalanceamento | Reamostragem (SMOTE) ou pesos de classe, aplicados só dentro de cada dobra da validação cruzada |
| 3. Mineração de dados | Treinar os algoritmos do Quadro 4 (GB, XGBoost, DT, KNN, RF, NB, RL, MLP) com GridSearchCV |
| 4. Avaliação | Validação cruzada k = 5; F1-score (principal), acurácia, precisão, recall e AUROC; curvas ROC |
| 5. Interpretação | Valores SHAP do melhor modelo |

## Decisões importantes até agora

- **Formato:** tabela plana, 1 linha por recém-nascido.
- **Todas as 99 colunas foram mantidas;** colapsar (por exemplo, juntar `_pre` e `_pos24h`) fica para depois, se necessário.
- **`dias_ate_dx_sepse` sai das variáveis de entrada.** Só existe para quem teve sepse, então prevê o desfecho por
  construção (vazamento de alvo).
- **A "zona cinzenta" fica:** `dias_utin`, `vm_dias`, `cvc_dias`, `npt_dias` e `hemocultura_*` seguem como preditoras,
  mas podem ser consequência da sepse. A recomendação para os dados reais é contar esses dias **até o momento da
  suspeita**, e não até a última evolução.
- **O conjunto de teste só é aberto no fim,** e tudo o que é "aprendido" dos dados (medianas, escalas) vem só do treino.
- **Ausentes dos exames:** preenchidos pela mediana do treino; os exames com 20% ou mais de ausentes ganham também uma
  coluna "foi medido?". Na hemocultura, o vazio de uma cultura negativa vira "não se aplica", e não "dado perdido".
- **Exames de cauda longa** (PCR, PCT, lactato, leucócitos etc.) passam por `log1p` antes da padronização.
- **Resultado no seed:** a matriz do treino fica com 800 linhas × 132 colunas, sem ausentes.

## Como usar

Requer Python 3.11 ou superior. Os comandos abaixo são para o Windows (PowerShell), a partir da pasta do projeto.

```powershell
# Instalar as dependências (uma vez)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Gerar os dados sintéticos -> data/seed_sepse.csv
.venv\Scripts\python.exe gerador_seed.py --n 1000 --prevalencia 0.20 --sinal 1.0 --ausentes 1.0 --seed 42

# 2. Auditar, dividir em treino e teste (-> data/treino.csv e data/teste.csv) e ajustar o pré-processador no treino
.venv\Scripts\python.exe preprocessamento.py --entrada data/seed_sepse.csv --seed 42

# Rodar os testes
.venv\Scripts\python.exe -m pytest
```

Parâmetros do gerador: `--sinal` (0 = classes indistinguíveis, 1 = efeito completo), `--ausentes` (multiplicador das
taxas de dados ausentes) e `--seed` (mesma semente gera exatamente o mesmo CSV).

## Estrutura

```
gerador_seed.py            gerador de dados sintéticos
preprocessamento.py        auditoria, divisão treino/teste e pré-processador (sklearn Pipeline, sem ajuste)
tests/                     testes automáticos (pytest)
plans/                     planos versionados, um por etapa
data/                      CSVs gerados (fora do git)
Projeto_sepse_07_07.pdf    projeto de pesquisa (fonte dos requisitos)
requirements.txt           dependências
```

## Limitações

- Os pesos e efeitos do gerador são palpites clínicos, não estimados de dados reais; os modelos "descobrem" o que foi
  plantado nos dados.
- No dado real, os efeitos serão mais complexos e o desempenho deve ser pior.
- Com ~700 pacientes e 15 a 25% de sepse (cerca de 105 a 175 casos) e 97 preditoras, há risco de sobreajuste; o projeto
  não prevê seleção de variáveis, ponto a discutir com a orientadora.
