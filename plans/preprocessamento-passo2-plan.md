# Plano de implementação: passo 2 (dividir treino e teste) do pré-processamento

> **Reconstruído** em 2026-10-28 a partir do plano geral (`preprocessamento-plan.md`), do código e do commit `98d2266`.
> Não é o texto original aprovado: o passo foi aprovado dentro do plano geral, sem arquivo próprio. Os desvios e os
> números do seed são os que constam no plano geral (subseção 5.1).
> O antigo plano geral (`preprocessamento-plan.md`) foi removido de `plans/` em 2026-10-28, a pedido do usuário (um arquivo
> por passo); continua no histórico do git, ex.: `git show 8cc96a9:plans/preprocessamento-plan.md`.

Passo anterior:
[`preprocessamento-passo1-plan.md`](preprocessamento-passo1-plan.md); próximo:
[`preprocessamento-passo3-plan.md`](preprocessamento-passo3-plan.md).

| | |
|---|---|
| **Status** | **Implementado** e commitado (`98d2266`, junto com o passo 1) |
| **Branch** | `create-pipeline` |
| **Código** | `preprocessamento.py` (`dividir`, `separar_x_y`, gravação pelo CLI), `tests/test_preprocessamento.py` (4 testes) |
| **Fonte no PDF** | Divisão 80/20 (p. 45) |

## 1. Contexto e objetivo

Antes de qualquer transformação que **aprenda** algo dos dados, é preciso separar o conjunto de teste, que só será aberto
uma vez, no fim. Este passo entrega:

1. `dividir(df, teste=0.20, seed=42)`: divisão 80/20 estratificada pelo desfecho.
2. `separar_x_y(df)`: separa as variáveis de entrada (X) do alvo (y), tirando o identificador e `dias_ate_dx_sepse`.
3. Gravação de `data/treino.csv` e `data/teste.csv` pelo CLI, para a divisão ficar congelada e auditável.

Fora do escopo: qualquer transformação dos valores (passo 3), validação cruzada e reamostragem (etapas seguintes).

## 2. Conceitos (para ler antes de aprovar)

**C1. Vazamento de dados no pré-processamento.** Qualquer valor aprendido dos dados (mediana usada para imputar, média e
desvio da padronização, categorias do one-hot, SMOTE) deve vir **só do treino**. Se a mediana for calculada com o conjunto
todo, o teste "vaza" informação para o treino e a métrica final fica otimista. Por isso a divisão vem **antes** de tudo, e o
pré-processador (passo 3) é entregue sem ajuste.

**C2. Divisão antes de tudo, estratificada.** 80% treino / 20% teste (p. 45), preservando a proporção de sepse em cada
parte. Com 1000 pacientes e 20% de prevalência: 800 treino (160 positivos) e 200 teste (40 positivos). O teste só é aberto
uma vez, no fim. (A p. 41 cita 70/30 para o *teste piloto*; para a validação do modelo vale a p. 45, 80/20.)

## 3. Decisões

| # | Decisão | Escolha | Origem |
|---|---|---|---|
| P1 | Vazamento de alvo | `separar_x_y` sempre tira `dias_ate_dx_sepse` de X. As demais colunas da "zona cinzenta" ficam | Usuário |
| P5 | Divisão | 80/20 estratificada, `random_state` fixo (padrão 42), gravada em `data/treino.csv` e `data/teste.csv` (pasta `data/` fora do git) | PDF |
| P8 | Dependência nova | `scikit-learn` (1.9.1 instala no Python 3.14; verificado com `pip install --dry-run`), adicionado ao `requirements.txt` | Proposta |
| P9 | `id_paciente` e `sepse_tardia` | Fora das variáveis de entrada (X); o alvo vira y | Proposta |

## 4. Desenho

```
df ─► dividir (80/20, estratificado) ─► treino (DataFrame completo) ─► separar_x_y ─► X_treino, y_treino
                                    └► teste  (DataFrame completo) ─► guardado; só aberto na avaliação final
```

- **`dividir(df, teste=0.20, seed=42)`** usa `train_test_split(..., stratify=df[COL_ALVO], random_state=seed)` e devolve os
  dois **DataFrames completos** (com `id_paciente` e alvo), com índice reiniciado, para a divisão poder ser gravada e
  auditada.
- **`separar_x_y(df)`** devolve `X` (sem `id_paciente`, `sepse_tardia` e `dias_ate_dx_sepse`, com `errors="ignore"`) e `y`.
- **CLI:** `python preprocessamento.py --entrada data/seed_sepse.csv --seed 42` (opções `--teste`, `--saida-dir`, `--seed`)
  imprime a auditoria (passo 1), grava os dois CSV e mostra, por parte, linhas e casos de sepse, além do número de variáveis
  de entrada.

## 5. Testes (4, em `tests/test_preprocessamento.py`)

1. Divisão 80/20 estratificada: 800 treino e 200 teste, com 160 e 40 positivos.
2. A divisão não repete nem perde pacientes (conjuntos de `id_paciente` disjuntos e com união igual ao original).
3. Reprodutível com a mesma semente e diferente com outra.
4. `separar_x_y` tira id, alvo e `dias_ate_dx_sepse`, e `y` é o alvo original.

## 6. Verificação e resultado no seed 42

`.venv\Scripts\python.exe preprocessamento.py` e `.venv\Scripts\python.exe -m pytest -q`. Resultado registrado no plano
geral: 800 treino (160 positivos) e 200 teste (40 positivos), ambos com 20,0%; X com 96 variáveis (97 menos
`dias_ate_dx_sepse`); 28 testes no total após este passo (17 do gerador, 7 do passo 1 e 4 deste).

## 7. Desvios entre o plano geral e a implementação

- **`dividir` devolve DataFrames completos.** O plano geral (4.2) previa `dividir -> X_treino, X_teste, y_treino, y_teste`.
  Ficou assim para a divisão poder ser gravada com `id_paciente` e alvo e auditada; a separação em X e y passou para a
  função à parte `separar_x_y`.
- `scikit-learn 1.9.1` foi instalado e adicionado ao `requirements.txt` neste passo.

## 8. Riscos e observações

- **Teste pequeno:** 200 linhas e cerca de 40 positivos dão métricas com intervalo de confiança largo; a comparação entre
  modelos deve vir principalmente da validação cruzada no treino.
- **O teste só é aberto no fim.** O CLI nunca o transforma; os testes automáticos é que verificam o `transform` dele.
- **Estratificação só pelo desfecho:** não garante equilíbrio de outras variáveis (idade gestacional, por exemplo).
- **Divisão por paciente (1 linha por recém-nascido):** não há risco de o mesmo paciente aparecer nos dois conjuntos; se
  o dado real vier com mais de uma linha por paciente, a divisão precisará ser por grupo.
