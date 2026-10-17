# 04 — Pontos de atenção no projeto

Observações levantadas na leitura do `Projeto_sepse_07_07.pdf` que afetam o pipeline. **Não são decisões** — são itens para discutir com a orientação.

## Possível vazamento de dados (data leakage)
Algumas variáveis listadas só existem **depois** (ou por causa) do desfecho:
- **`dias_ate_diagnostico`** (Quadro 2): só existe para quem teve sepse → praticamente revela o alvo.
- **`dias_utin`** (tempo total na UTIN): conhecido só na alta; sepse prolonga a internação.
- **Resultado da hemocultura / perfil de sensibilidade** (Quadro 3): hemocultura positiva é parte do diagnóstico de sepse.
- **"Antibióticos prévios à sepse tardia"**: não há "prévio à sepse" para quem não teve sepse — precisa de uma referência temporal comum.
- Regra da p. 38 ("dados da **última evolução**") vs. janela em torno da suspeita (p. 40): a última evolução pode ser posterior ao diagnóstico.

## Referência temporal para os não-sépticos
A janela dos exames é centrada em T0 = suspeita (coleta de hemocultura). Para pacientes que nunca tiveram suspeita, é preciso definir um T0 equivalente (ex.: dia de internação pareado, ou toda suspeita com hemocultura — positiva ou negativa — vira uma amostra).

## Escores
- **nSOFA** aparece nas pp. 43 e 46, mas **não** está no Quadro 3. O Quadro 3 tem Glasgow neonatal adaptada.
- **NEWS2 e qSOFA** são escores de **adultos**; aplicação em neonatos exige adaptação de pontos de corte (definir referência).

## Divisões de dados
- Teste piloto: holdout 70/30 (p. 41). Modelagem: 80/20 + k-fold k=5 (p. 45). Definir se o k-fold roda dentro dos 80% (recomendado) e o 20% fica intocado até o fim.
- Com ~700 pacientes e 15–25% de positivos, o teste terá ~20–35 positivos → usar divisão **estratificada** (`StratifiedKFold`, `stratify=y`).

## Dados ausentes
p. 38 diz que prontuários com dados incompletos em **qualquer** variável serão excluídos; pp. 40 e 44 falam em **imputação**. Excluir tudo pode derrubar muito a amostra (ex.: PCT "quando disponível").

## Variáveis não especificadas
- Sinais vitais (FC, FR, SpO2, PA): não diz qual agregação (valor em T0? mín/máx/média?). Só temperatura tem mín/máx/média.
- "Classes de antibióticos" e "agente isolado" são multi-categóricas — definir codificação.

## Outros
- "Regressão" é citada como tarefa, mas o desfecho é binário — na prática, apenas classificação.
- Corte de "70% nas métricas" para DNN/SVM: definir em qual métrica (F1? AUROC?).
