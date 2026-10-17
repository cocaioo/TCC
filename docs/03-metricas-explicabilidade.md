# 03 — Métricas de avaliação e explicabilidade

> Fonte: seção 3.9, Quadro 5 e Figura 3 (pp. 45–46) do `Projeto_sepse_07_07.pdf`.

## Quadro 5 — Métricas de desempenho (p. 45)

| Métrica | Definição | Papel no estudo | scikit-learn |
|---|---|---|---|
| **F1-score** | 2 · (precisão · recall) / (precisão + recall), 0–1 | **Métrica principal** (desbalanceamento esperado) | `f1_score`, `scoring="f1"` |
| Acurácia | Proporção de previsões corretas | Complementar | `accuracy_score` |
| Precisão | VP / (VP + FP) — não gerar falsos positivos | Complementar | `precision_score` |
| Recall (Sensibilidade) | VP / (VP + FN) — identificar todos os casos de sepse | Complementar | `recall_score` |
| **AUROC** | Área sob a curva ROC (TPR × FPR); quanto mais próximo de 1, melhor | Co-principal (Figura 3: "F1-score · AUROC") | `roc_auc_score`, `scoring="roc_auc"` |

**Justificativa (p. 46):** reduzir falsos positivos (gravidade da condição) e falsos negativos (evitar uso inadequado do modelo na UTIN).

**Apresentação:** tabela comparativa entre algoritmos + **curvas ROC de todos os modelos** em um gráfico.

**Critério de corte:** algoritmos extras (DNN, SVM) são excluídos se < 70% nas métricas (p. 45).

## Explicabilidade — SHAP (p. 46)
- Aplicada **somente ao modelo de melhor desempenho global**.
- Objetivo: identificar as variáveis que mais contribuem **para cada predição individual**, tornando o modelo compreensível para a equipe da UTIN.
- Produto final (Figura 3): "Modelo preditivo interpretável para a UTIN — apoio à decisão clínica na sepse neonatal tardia".

### Saídas SHAP sugeridas para o pipeline
| Nível | Gráfico | Função (`shap`) |
|---|---|---|
| Global | Importância média \|SHAP\| | `shap.plots.bar` |
| Global | Distribuição por variável | `shap.plots.beeswarm` |
| Efeito de uma variável | Dependência | `shap.plots.scatter` |
| Individual (paciente) | Contribuição por variável | `shap.plots.waterfall` / `shap.plots.force` |

Explainer conforme o modelo vencedor: `TreeExplainer` (DT, RF, GB, XGBoost), `LinearExplainer` (RL), `KernelExplainer`/`Explainer` genérico (KNN, NB, MLP, SVM — mais lento).
