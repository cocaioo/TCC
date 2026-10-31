# Predição de sepse neonatal tardia com aprendizado de máquina

Pipeline de aprendizado de máquina para estimar a probabilidade de **sepse neonatal tardia** (CID-10 A41.9 / P36) em
recém-nascidos internados em Unidade de Terapia Intensiva Neonatal (UTIN), a partir de dados do prontuário eletrônico
(SOUL MV). O desenho do estudo está em `Projeto_sepse_07_07.pdf` e segue o processo KDD (Descoberta de Conhecimento em
Bases de Dados).

## Escopo atual

O repositório contém só a parte do pipeline que **não depende dos dados**: a divisão e a validação cruzada
estratificadas, o catálogo dos 8 algoritmos do Quadro 4, as grades de hiperparâmetros, o tratamento do desbalanceamento,
as métricas do Quadro 5, a avaliação final e a explicação com SHAP.

Não há dados no repositório, nem reais nem sintéticos. Os dados reais dependem de aprovação em comitê de ética, do
conector com o prontuário e de definições de desenho ainda em aberto (ver `docs/entendimento-do-projeto.md`). O
**pré-processamento** (limpeza, valores ausentes, codificação e escolha das variáveis) depende desses dados e ainda não
existe: o pipeline tem um espaço reservado para ele, e até lá espera uma tabela já numérica e sem ausentes.

O relatório técnico com o que foi desenvolvido está em `docs/relatorio-tecnico.pdf` (fonte em `docs/relatorio-tecnico.tex`).

## Módulos

| Módulo | O que faz |
|---|---|
| `divisao.py` | Lê a tabela (uma linha por recém-nascido, identificador `id_prontuario` e desfecho `sepse_tardia` 0/1) e divide em treino (80%) e teste (20%), estratificado pelo desfecho |
| `modelos.py` | Catálogo dos 8 algoritmos (RL, NB, KNN, DT, RF, GB, XGBoost, MLP), duas grades de hiperparâmetros por algoritmo, três modos de desbalanceamento (nenhum, pesos de classe, SMOTE) e montagem dos pipelines |
| `validacao.py` | Busca em grade (`GridSearchCV`) e validação cruzada estratificada com 5 dobras, simples e aninhada, só com o treino |
| `avaliacao.py` | Escolhe o melhor modelo pela validação, abre o teste uma única vez, calcula intervalos de 95% por bootstrap e gera curvas ROC e matriz de confusão |
| `explicacao.py` | Valores SHAP do melhor modelo, modelo final salvo (pipeline e descrição em JSON) e função `prever` |

## Princípios metodológicos

- O conjunto de teste permanece lacrado até a avaliação final e é aberto uma única vez.
- Tudo o que é aprendido dos dados (pré-processamento, pesos de classe, exemplos do SMOTE) vem apenas do treino de cada
  rodada; a dobra de validação não influencia o ajuste. Isso é coberto por testes que falham se o vazamento reaparecer.
- O desempenho é estimado de duas formas: busca simples (nota otimista do vencedor) e validação cruzada aninhada (nota
  honesta), que é a usada para comparar os algoritmos.
- O F1-score (limiar 0,5) é a métrica principal de seleção, conforme o projeto; acurácia, precisão, sensibilidade,
  especificidade e AUROC também são registradas.
- Nos modelos sem suporte a pesos de classe no scikit-learn (KNN, NB e MLP), esse modo não aplica ajuste; o SMOTE cobre
  os 8 modelos e roda só no treino de cada ajuste.

## Como executar

Requer Python 3.11 ou superior. Os comandos são para o Windows (PowerShell), a partir da pasta do projeto.

```powershell
# Instalar as dependências (uma vez)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Listar o catálogo de modelos (combinações das grades e ajuste de desbalanceamento de cada algoritmo)
.venv\Scripts\python.exe modelos.py --listar

# Testes automáticos
.venv\Scripts\python.exe -m pytest
```

Os testes usam uma tabela numérica genérica, criada na memória por `tests/dados_genericos.py`, sem nenhum significado
clínico. Ela serve só para verificar a mecânica do pipeline; nada é gravado e nenhum número dela é resultado do estudo.

Quando houver dados reais e pré-processamento, a sequência de execução será: `divisao.py` (gera treino e teste),
`validacao.py` (uma vez por modo de desbalanceamento), `avaliacao.py` e `explicacao.py`. Cada um tem `--help`.

## Estrutura

```
divisao.py                 leitura da tabela e divisão treino/teste
modelos.py                 catálogo dos 8 algoritmos, grades de hiperparâmetros e montagem dos pipelines
validacao.py               busca em grade e validação cruzada (simples e aninhada)
avaliacao.py               avaliação final no teste, curvas ROC e matriz de confusão
explicacao.py              valores SHAP do vencedor, modelo final salvo e função prever
tests/                     testes automáticos (pytest), um arquivo por módulo
docs/                      entendimento do projeto, relatório técnico e material de estudo
Projeto_sepse_07_07.pdf    projeto de pesquisa
requirements.txt           dependências
```

A pasta `data/` está no `.gitignore`: os dados reais serão privados e nunca vão para o git.
