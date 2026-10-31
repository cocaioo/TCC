# Entendimento do projeto: predição de sepse neonatal tardia

Documento de alinhamento entre a equipe do estudo. Consolida a leitura do projeto de pesquisa
(`Projeto_sepse_07_07.pdf`) que orienta o desenvolvimento do pipeline de aprendizado de máquina, e registra os pontos que
precisam de definição. É um documento vivo: cada seção é acrescentada à medida que o entendimento é discutido e validado.

As páginas citadas são as numeradas no próprio PDF.

## 1. Objetivo do estudo

O objetivo é **desenvolver e avaliar algoritmos de inteligência artificial para a predição de sepse tardia em
recém-nascidos internados em UTIN** de maternidade pública de referência em alta complexidade no Nordeste do Brasil
(Resumo; objetivo geral na seção 1.6.1, p. 16).

A ênfase está na **avaliação**. A questão de pesquisa (seção 1.3, p. 14) pergunta se algoritmos treinados com dados de
prontuário eletrônico são capazes de predizer a ocorrência de sepse tardia com desempenho adequado.

### 1.1 O que "avaliar" significa no projeto

Conforme as seções 3.8 e 3.9 (p. 44 a 46):

1. Treinar os algoritmos do Quadro 4 (p. 44): Regressão Logística, Naive Bayes, KNN, Árvore de Decisão, Random Forest,
   Gradient Boosting, XGBoost e Redes Neurais Artificiais (MLP).
2. Medir todos com as mesmas métricas do Quadro 5 (p. 45): F1-score como métrica principal, além de acurácia, precisão,
   recall e AUROC.
3. Estimar o desempenho por validação cruzada k-fold com k = 5, com busca de hiperparâmetros por `GridSearchCV` e
   separação de 80% dos dados para treino e 20% para teste.
4. Comparar os algoritmos em tabela e graficamente, com curvas ROC de todos os modelos.
5. Submeter o modelo de melhor desempenho à análise de explicabilidade com valores SHAP.

### 1.2 Protótipo de sistema de apoio à decisão

O projeto não prevê o desenvolvimento nem a integração de um sistema de apoio à decisão. O Resumo apresenta como
resultados esperados a identificação das variáveis preditoras de maior relevância e a construção de modelo preditivo de
bom desempenho, "com vistas à proposição" de um protótipo. O objetivo específico correspondente (seção 1.6.2, p. 16) usa
o verbo **propor**.

Entendimento adotado: o produto do estudo é o modelo avaliado, construído de forma que possa vir a ser integrado a um
sistema de apoio (um único pipeline com pré-processamento e modelo, lista fixa de variáveis de entrada e saída em
probabilidade). A proposta do protótipo é tratada como descrição na tese, e não como implementação.

## 2. Os dados

### 2.1 População e amostra

- **Local:** UTIN da Nova Maternidade Dona Evangelina Rosa (NMDER), Teresina, com 30 leitos (seção 3.2, p. 37).
- **Período:** dados retrospectivos de 2023 a 2025.
- **Amostra estimada:** no mínimo 700 recém-nascidos, com incidência esperada de sepse tardia entre 15% e 25%.
- **Inclusão** (seção 3.3, p. 37): internação na UTIN por mais de 72 horas; idade gestacional igual ou superior a 22
  semanas; dados registrados em prontuário eletrônico; ausência de diagnóstico de sepse precoce nos três primeiros dias.
- **Exclusão** (p. 37 e 38): malformações congênitas graves ou síndromes genéticas que comprometam a resposta imune;
  óbito ou transferência antes de 72 horas de internação; dados incompletos para as variáveis preditoras essenciais.

### 2.2 Variável-alvo

Ocorrência de sepse neonatal tardia durante a internação, variável dicotômica (sim ou não), avaliada na evolução médica a
partir dos códigos CID-10 A41.9 (septicemia não especificada) e P36 (septicemia bacteriana do recém-nascido) (seção 3.4,
p. 38).

### 2.3 Variáveis preditoras

Organizadas em cinco categorias (seção 3.5, p. 38 a 40):

| Categoria | Fonte no PDF | Exemplos |
|---|---|---|
| Demográficas e do parto | Quadro 2 (p. 39) | sexo, idade gestacional, peso ao nascer, Apgar, tipo de parto, intercorrências gestacionais |
| Clínicas e de internação | Quadro 2 (p. 39) | ventilação mecânica, cateteres, nutrição parenteral, sinais vitais, vasopressor |
| Laboratoriais | Quadro 3 (p. 40) | leucócitos, relação I/T, plaquetas, PCR, procalcitonina, lactato |
| Microbiológicas | Quadro 3 (p. 40) | hemocultura, perfil de sensibilidade, colonização |
| Escores clínicos | Quadro 3 (p. 40) | Glasgow neonatal adaptada, NEWS2, qSOFA |

### 2.4 Forma da base

- **Uma linha por recém-nascido**, conforme a ficha de coleta do apêndice (p. 65 a 67).
- Sinais vitais e evolução clínica: registros da última evolução diária documentada (seção 3.7, p. 43).
- Exames seriados: a última medida disponível antes da suspeita clínica de sepse e as primeiras medidas obtidas até 24
  horas depois (seção 3.5.2, p. 39 e 40).
- Extração do prontuário SOUL MV via API, com exportação em CSV e identificação sequencial anonimizada (seção 3.7, p. 42
  e 43).

## 3. Situação do desenvolvimento

Os dados reais dependem da aprovação no Comitê de Ética e do acesso à API. Até lá, o repositório contém só a parte do
pipeline que **não depende dos dados**: ela recebe uma tabela já pronta e não trata nem transforma os dados. O
pré-processamento (limpeza, ausentes, codificação e escolha das variáveis) será definido com os dados reais e com as
respostas aos pontos da seção 4.

| Requisito (seção 1.1) | Situação |
|---|---|
| Divisão 80/20, estratificada pelo desfecho | Implementado |
| Os 8 algoritmos do Quadro 4 | Implementados |
| `GridSearchCV` e validação cruzada com k = 5 | Implementados (validação simples e aninhada) |
| Pesos de classe e reamostragem sintética (SMOTE) | Implementados |
| Métricas do Quadro 5 | Implementadas |
| Avaliação final no conjunto de teste e curvas ROC | Implementada, nunca executada com dados reais |
| Explicabilidade com SHAP | Implementada, nunca executada com dados reais |
| Pré-processamento | A definir |

O relatório técnico, em `docs/relatorio-tecnico.pdf`, descreve como cada requisito foi implementado.

## 4. Pontos a definir com a orientação

### 4.1 Referência temporal dos exames para quem não teve sepse

A seção 3.5.2 (p. 39 e 40) define como referência temporal dos exames laboratoriais "o momento da suspeita clínica de
sepse tardia, isto é, o instante em que a equipe assistencial [...] solicitou a coleta de hemocultura", e considera a
última medida antes desse marco e as primeiras até 24 horas depois.

Para recém-nascidos que tiveram suspeita, esse marco existe. Para recém-nascidos que nunca tiveram suspeita nem
hemocultura colhida, o projeto não define qual exame utilizar.

**Por que importa:** se os casos têm exames colhidos no momento em que estavam doentes e os controles têm exames de um
momento qualquer da internação, o modelo tende a aprender a reconhecer quem já estava em investigação, e não a predizer a
sepse. O desempenho pode parecer alto sem refletir capacidade preditiva real.

