# Como rodar o projeto

Guia rápido de execução. Para entender o que cada etapa faz e o que os números
significam, ver o [README](README.md).

Requer **Python 3.11+**. Todos os comandos rodam **a partir da raiz do projeto**
(`projeto-cripto/`).

---

## 1. Ambiente

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No Linux/macOS, a ativação é `source .venv/bin/activate`.

Confira que o ambiente está ativo — o prompt do terminal deve começar com `(.venv)`:

```powershell
python -c "import pandas, sklearn, matplotlib; print('ok')"
```

## 2. Pipeline completo

```powershell
$env:USAR_SNAPSHOT = "1"      # Linux/macOS: export USAR_SNAPSHOT=1

python src/collect.py
python src/preprocess.py
python src/features.py
python src/clustering.py
python src/classification.py
python src/regression.py
python src/prever.py
```

A ordem importa: cada script lê o Parquet que o anterior gravou. Ao final,
`data/processed/` tem as tabelas e `reports/` tem os relatórios `.md`.

**Por que `USAR_SNAPSHOT=1`:** a API da CoinGecko devolve uma janela *móvel* de 365
dias, então coletar em outra data gera outra série e os números dos relatórios deixam
de bater. O snapshot congelado (`data/snapshot/solana_365d.csv`, 2025-09-01 a
2026-08-31) é o que reproduz exatamente os resultados entregues. Sem a variável, o
`collect.py` busca dados novos na API — é o diferencial de "atualização automática",
não o caminho da reprodução.

## 3. Verificação de vazamento temporal

```powershell
python tests/test_vazamento.py
```

São 8 verificações de que os atributos usam apenas dados até o instante `t`. É o
argumento metodológico central do trabalho — vale rodar antes de apresentar.

## 4. Análise exploratória e painel

```powershell
jupyter notebook notebooks/eda_solana.ipynb    # EDA, 7 visualizações
streamlit run app/dashboard.py                 # painel interativo
```

O painel depende do pipeline já ter rodado (passo 2).

---

## Problemas comuns

**`ModuleNotFoundError: No module named 'pandas'` (ou `matplotlib`)**

O Python em uso não é o do `.venv`. Causa mais frequente: o botão ▶ do VS Code, ou a
extensão Code Runner, chamando o `python` do PATH global. Duas saídas:

- No VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → escolher o de
  `.\.venv\Scripts\python.exe`, e recarregar a janela. O arquivo
  `.vscode/settings.json` já deixa isso configurado no repositório.
- No terminal: ativar o venv (passo 1) ou chamar o executável direto —
  `.venv\Scripts\python.exe src\collect.py`.

**`Arquivo de entrada não encontrado: ...precos.parquet`** (ou `atributos.parquet`)

Os Parquet de `data/processed/` não são versionados (estão no `.gitignore`), porque
são regeráveis. Rode o pipeline do passo 2 na ordem, começando pelo `collect.py`.

**Erro de rede / HTTP 429 no `collect.py`**

Você está batendo na API sem o snapshot. Defina `USAR_SNAPSHOT=1` e rode de novo. Se
quiser mesmo coletar dados novos, uma chave Demo da CoinGecko na variável
`COINGECKO_API_KEY` eleva o limite para 100 requisições/minuto.
