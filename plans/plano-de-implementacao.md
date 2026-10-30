# Plano de implementação

Plano único do pipeline, que substitui os planos por passo anteriores (eles permanecem no histórico do git). Resulta da
leitura do projeto de pesquisa (`Projeto_sepse_07_07.pdf`) registrada em `docs/entendimento-do-projeto.md`.

| | |
|---|---|
| **Status** | Aprovado em 2026-10-29; em execução |
| **Branch** | `create-pipeline` |

## 1. Objetivo e premissa

O estudo desenvolve e avalia algoritmos de aprendizado de máquina para a predição de sepse tardia em UTIN (Resumo;
seção 1.6.1). Este plano conclui o pipeline que **testa os algoritmos de ponta a ponta**, da tabela de entrada até a
explicação do melhor modelo, com os dados sintéticos.

Premissa: só se implementa o que **não depende dos dados reais**. A forma dos dados reais depende de definições ainda em
aberto com a orientação, principalmente se o modelo responde à pergunta "quem vai ter sepse" (alerta antes da suspeita) ou
"este recém-nascido investigado tem sepse" (apoio na suspeita); ver `docs/entendimento-do-projeto.md`. O que depende dos
dados fica **incompleto de propósito, mas isolado**: as listas de colunas no topo de `preprocessamento.py`. Quando os dados
chegarem, o ajuste se concentra ali.

## 2. Situação de partida

| # | Estação | Situação | Neste plano |
|---|---|---|---|
| 1 | Dados de entrada (`gerador_seed.py`) | Pronta (sintéticos) | Sem mudança |
| 2 | Pré-processamento (`preprocessamento.py`) | Pronta; parte específica dos dados isolada no topo | Sem mudança |
| 3 | Divisão treino/teste 80/20 | Pronta | Sem mudança |
| 4 | Treino e comparação (`modelos.py`, `validacao.py`) | Pronta, falta o SMOTE | Passo 2 |
| 5 | Avaliação final | Não existe | Passo 3 |
| 6 | Explicação e modelo salvo | Não existe | Passo 4 |

Passos: 1 README; 2 SMOTE; 3 avaliação final; 4 SHAP e modelo salvo; 5 relatório do que foi feito.

Princípios mantidos de todo o trabalho anterior: o teste é aberto uma única vez, no passo 3; tudo o que é aprendido dos
dados vem só do treino de cada rodada; cada passo traz testes automáticos que falham se o princípio for violado.

## 3. Passo 1: README

**O que é.** Reescrever o README para refletir a premissa da seção 1.

**Conteúdo.**
- O que o pipeline faz (testa os 8 algoritmos de ponta a ponta) e com quais dados (sintéticos, sem valor científico).
- As seis estações e o estado de cada uma.
- Onde fica o que depende dos dados reais (as constantes do topo de `preprocessamento.py` e o gerador) e o que se espera
  ajustar quando eles chegarem.
- Ligação para `docs/entendimento-do-projeto.md` (requisitos e pontos em aberto) e para este plano.
- Remoção da tabela de "passos e planos", que apontava para os planos antigos.

**Mudanças nos arquivos.** `README.md`; remoção dos 8 arquivos `plans/*-passoN-plan.md`.

## 4. Passo 2: SMOTE (fecha a estação 4)

**O que é.** Terceiro modo de lidar com o desbalanceamento, ao lado de `nenhum` e `class_weight`. O PDF prevê avaliar
"técnicas de reamostragem sintética e ajuste de pesos de classe" (seção 3.9, p. 46). O SMOTE cria exemplos sintéticos de
sepse por interpolação entre vizinhos e, por agir nos dados, também cobre KNN, NB e MLP, que não têm `class_weight`.

**Decisões.**

| # | Decisão | Motivo |
|---|---|---|
| S1 | `DESBALANCEAMENTO = ("nenhum", "class_weight", "smote")`; no modo `smote`, os estimadores são os do modo `nenhum` | Não corrigir o desbalanceamento duas vezes |
| S2 | `SMOTE` simples do `imbalanced-learn` (`k_neighbors=5`, classes igualadas, `random_state=seed`) | Variantes (SMOTE-NC e outras) ficam para depois, se necessário |
| S3 | `construir_pipeline(prep, estimador, desbalanceamento="nenhum", seed=42)` devolve sempre um `Pipeline` do `imbalanced-learn` com os passos do pré-processador "achatados": `[estrutural, log1p, colunas, float, (smote), modelo]` | O `imbalanced-learn` recusa um `Pipeline` dentro de outro; um formato único para os três modos |
| S4 | O SMOTE só roda no `fit` (treino de cada rodada), nunca na previsão | Garantido pelo `Pipeline` do `imbalanced-learn`; reamostrar antes de dividir vazaria informação |
| S5 | O interruptor `excluir_sinalizadas` do plano anterior **sai do escopo** | Quais colunas excluir depende da definição da pergunta (seção 1) |

**Mudanças nos arquivos.** `modelos.py` (modo, pipeline achatado, `imprimir_catalogo`); `validacao.py` (repassa
`desbalanceamento` e `seed` a `construir_pipeline`); testes que usavam `pipe["prep"]` passam a usar os passos achatados.

**Testes novos.**
1. No modo `smote`, os estimadores são iguais aos do modo `nenhum`.
2. O passo `smote` fica entre `float` e `modelo` nesse modo e não existe nos outros.
3. Os 8 pipelines ajustam e devolvem probabilidades válidas no modo `smote`.
4. Só no treino: um estimador espião recebe classes igualadas no `fit` e exatamente as linhas pedidas no `predict_proba`.
5. Sem vazamento: com alvo embaralhado, a AUROC fora da dobra com o SMOTE dentro do pipeline fica perto de 0,5; o controle
   (reamostrar antes de dividir) dá AUROC claramente maior, o que prova que o teste detectaria o vazamento.
6. Mesma semente, mesmo resultado.
7. `comparar_modelos(desbalanceamento="smote")` roda e mantém o alinhamento das probabilidades fora da dobra.

**Limitações registradas.** O SMOTE gera valores fracionários em colunas 0/1 (por exemplo, sexo = 0,3); árvores lidam bem,
KNN e MLP podem ser mais sensíveis. No MLP com parada precoce, a validação interna sai dos dados já reamostrados.

## 5. Passo 3: avaliação final (estação 5)

**O que é.** Abrir o conjunto de teste (20%) uma única vez e produzir a tabela e as curvas ROC que o PDF pede (seção 3.9,
p. 46: "comparados entre si e apresentados em tabela e graficamente, com visualização das Curvas ROC para todos os
modelos").

**Decisões.**

| # | Decisão | Motivo |
|---|---|---|
| A1 | Especificidade (recall da classe negativa) entra nas métricas da validação cruzada e do teste | Citada no Resumo; complementa o recall (sensibilidade) |
| A2 | `validacao.py` passa a salvar, por modo, os 8 pipelines vencedores da busca simples, já reajustados no treino inteiro (`data/resultados/modelos_<grade>_<modo>/<modelo>.joblib`) | São os modelos finais de cada algoritmo; a avaliação só os carrega |
| A3 | O vencedor é escolhido **antes** de abrir o teste: maior F1 médio da validação aninhada entre todas as combinações modelo × modo disponíveis; empate desfeito pela AUROC | O teste não pode influenciar a escolha |
| A4 | Os 8 modelos de cada modo são medidos no teste e apresentados na tabela, com a ressalva de que a escolha já estava feita | O PDF pede a comparação de todos |
| A5 | Intervalo de confiança de 95% por bootstrap (1.000 reamostragens do teste) para F1 e AUROC do vencedor | Com cerca de 40 casos de sepse no teste, a nota pontual oscila muito |
| A6 | Limiar de decisão de 0,5; calibração das probabilidades fora do escopo | Ambos ficam como pontos a discutir com a orientação |

**Saídas** (em `data/resultados/`): tabela do teste (CSV); curvas ROC das probabilidades fora da dobra (treino) e do teste,
uma figura por modo com os 8 modelos (PNG); matriz de confusão do vencedor; resumo impresso no console.

**Mudanças nos arquivos.** Novo `avaliacao.py`; `validacao.py` (especificidade e salvamento dos modelos);
`requirements.txt` (`matplotlib`); novo `tests/test_avaliacao.py`.

**Testes novos.**
1. Especificidade conferida à mão num exemplo pequeno.
2. A escolha do vencedor não lê o arquivo de teste (espião em `carregar`).
3. A escolha segue F1 e, no empate, AUROC.
4. A tabela do teste tem as métricas do Quadro 5 mais a especificidade, uma linha por modelo e modo.
5. O intervalo do bootstrap contém a nota pontual e é reprodutível com a mesma semente.
6. As figuras são geradas (arquivo existe e não está vazio).
7. Um modelo salvo pela validação, carregado de volta, dá as mesmas probabilidades.

## 6. Passo 4: SHAP e modelo salvo (estação 6)

**O que é.** Explicar o vencedor com valores SHAP (seção 3.9, p. 46; objetivo específico 2) e deixar o modelo final salvo
de forma utilizável por outro programa, sem implementar nenhum sistema de apoio à decisão.

**Decisões.**

| # | Decisão | Motivo |
|---|---|---|
| E1 | `shap.Explainer` sobre a parte "modelo" do pipeline, com as variáveis já pré-processadas; fundo de até 100 linhas do treino; explicação das linhas do teste | O SHAP escolhe o algoritmo adequado (árvores, linear ou genérico) |
| E2 | Saídas: importância global (média do valor SHAP absoluto por variável, CSV), gráfico de resumo (beeswarm) e explicação de um caso individual (waterfall) | Global para o objetivo específico 2; individual para mostrar como seria a explicação de um alerta |
| E3 | Indicadores de ausência e colunas one-hot aparecem com o nome gerado pelo pré-processador | Reagrupar por variável original fica para quando as variáveis reais estiverem definidas |
| E4 | Modelo final: `data/resultados/modelo_final.joblib` (pipeline completo, do CSV bruto à probabilidade) e `modelo_final.json` (algoritmo, modo, hiperparâmetros, colunas de entrada esperadas, limiar, métricas, data) | Um único arquivo carregável, com o contrato de entrada documentado |
| E5 | Função `prever(df)`: confere as colunas de entrada contra o JSON, avisa o que falta ou sobra e devolve a probabilidade de sepse por paciente | Mostra que o modelo é integrável |

**Mudanças nos arquivos.** Novo `explicacao.py`; `requirements.txt` (`shap`); novo `tests/test_explicacao.py`.

**Testes novos.**
1. Para um modelo de árvore, a soma dos valores SHAP mais o valor-base reproduz a saída do modelo.
2. A importância global tem uma linha por variável da matriz pré-processada.
3. O modelo salvo e recarregado dá as mesmas probabilidades.
4. `prever` recusa uma tabela sem colunas obrigatórias e aceita colunas extras com aviso.
5. O JSON tem todos os campos e as colunas de entrada batem com as do treino.

## 7. Passo 5: relatório do que foi feito

**O que é.** Documento `docs/relatorio-do-pipeline.md`, uma espécie de README estendido para a orientação e a equipe do
estudo.

**Conteúdo.**
- As etapas, na ordem do processo KDD, com o que cada uma faz e **por que** foi feita daquela forma (decisões e
  alternativas descartadas), com referência ao PDF.
- Os resultados com os dados sintéticos (tabelas e figuras copiadas para `docs/figuras/`), com a ressalva de que não têm
  valor científico.
- As dúvidas que surgiram ao analisar o projeto e os dados, ligadas a `docs/entendimento-do-projeto.md`.
- O que será acrescentado ou ajustado quando os dados reais chegarem.

**Execução.** O pipeline inteiro é rodado de novo antes de escrever, para que todos os números do relatório venham da
mesma execução, com os comandos registrados no próprio documento.

## 8. Execução e verificação

Cada passo é implementado, testado (suíte completa com `pytest`), revisado de forma adversarial por um agente
independente e só então commitado. Ao final, uma execução ponta a ponta com os comandos do README deve rodar sem erro e
produzir todas as saídas.

## 9. Fora do escopo

Definição da pergunta (alerta antes da suspeita × apoio na suspeita) e da referência temporal dos exames; conector com o
SOUL MV; ajuste das listas de colunas para os dados reais; interruptor de colunas sinalizadas; calibração e escolha do
limiar; SVM e redes neurais profundas; análise estatística descritiva; nSOFA; qualquer interface ou sistema de apoio à
decisão.
