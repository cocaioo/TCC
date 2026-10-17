# TCC — Predição de sepse neonatal tardia (UTIN/NMDER)

O projeto completo está em `Projeto_sepse_07_07.pdf`, mas **consulte primeiro os resumos em `docs/`** (só abra o PDF se faltar algo):

- `docs/00-visao-geral.md` — estudo, população, alvo, fluxo KDD
- `docs/01-dados-variaveis.md` — variáveis (Quadros 2 e 3) e janela temporal
- `docs/02-algoritmos.md` — algoritmos (Quadro 4) e validação
- `docs/03-metricas-explicabilidade.md` — métricas (Quadro 5) e SHAP
- `docs/04-pontos-de-atencao.md` — riscos/inconsistências (leakage etc.)

Os dados reais são privados e ainda não chegaram; o pipeline deve funcionar com dados sintéticos até lá.

## Como rodar
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m sepse.dados_sinteticos --n 700 --prevalencia 0.2 --faltantes 0.1 --semente 42
```
- `src/sepse/schema.py` — fonte única das colunas (tipos, faixas, flag `suspeita_vazamento`).
- `src/sepse/io.py` — `ler_csv`/`salvar_csv` aplicando o schema (usar também para os dados reais).
- `src/sepse/dados_sinteticos.py` — gerador; sinal plantado em `SINAL_PLANTADO`.
- `data/` é ignorado pelo git (dados reais são privados).
