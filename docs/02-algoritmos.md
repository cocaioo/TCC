# 02 — Algoritmos e protocolo de validação

> Fonte: seções 3.7–3.9 e Quadro 4 (pp. 44–45) do `Projeto_sepse_07_07.pdf`.

## Quadro 4 — Algoritmos a serem testados (p. 44)

| Algoritmo | Sigla | Justificativa no projeto | Implementação provável |
|---|---|---|---|
| Gradient Boosting | GB | Combina árvores fracas; capta relações não lineares entre variáveis clínicas | `sklearn.ensemble.GradientBoostingClassifier` |
| XGBoost | — | GB com regularização e eficiência; muito usado em dados tabulares de saúde | `xgboost.XGBClassifier` |
| Árvore de Decisão | DT | Alta interpretabilidade para a equipe clínica | `sklearn.tree.DecisionTreeClassifier` |
| K-Nearest Neighbors | KNN | Classe majoritária dos vizinhos (distância euclidiana) | `sklearn.neighbors.KNeighborsClassifier` (exige escalonamento) |
| Random Forest | RF | Conjunto de árvores; robusto a outliers e ruído | `sklearn.ensemble.RandomForestClassifier` |
| Naive Bayes | NB | Teorema de Bayes com independência entre atributos; simples | `sklearn.naive_bayes.GaussianNB` |
| Regressão Logística | RL | **Modelo de referência (baseline)** | `sklearn.linear_model.LogisticRegression` (exige escalonamento) |
| Redes Neurais Artificiais | ANN/MLP | Relações não lineares complexas | `sklearn.neural_network.MLPClassifier` (exige escalonamento) |

**Extras citados na linha da ANN (pp. 44–45):** Redes Neurais Profundas e **SVM com kernels complexos** (`sklearn.svm.SVC`, kernel RBF/poly) também serão avaliados, **sendo excluídos se ficarem abaixo de 70% nas métricas avaliadas**.

## Protocolo de validação descrito
- **Divisão treino/teste:** `train_test_split` do scikit-learn, **80% / 20%** (p. 45).
  - O teste piloto usa holdout 70/30 (p. 41) — são etapas diferentes.
- **Otimização de hiperparâmetros:** `GridSearchCV` (`sklearn.model_selection`) (p. 45).
- **Validação cruzada:** k-fold com **k = 5**; métricas finais = média aritmética das 5 dobras; via `cross_val_score` (p. 45).
- Texto menciona "cruzamentos multivariados entre as variáveis" por serem dados não lineares (pp. 44–45) — sem detalhamento.

## Desbalanceamento (p. 46)
Avaliar **reamostragem sintética** (ex.: SMOTE, `imblearn`) e **ajuste de pesos de classe** (`class_weight="balanced"`, `scale_pos_weight` no XGBoost). Reamostragem deve ocorrer **apenas dentro do fold de treino** (usar `imblearn.pipeline.Pipeline`).

## Seleção do modelo
Comparar todos os algoritmos em tabela e graficamente (curvas ROC de todos). O melhor desempenho global vai para a análise SHAP — ver [03-metricas-explicabilidade.md](03-metricas-explicabilidade.md).
