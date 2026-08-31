# Sistema Inteligente de Mapeamento e Análise do Mercado de Criptomoedas

Projeto integrador da disciplina de Aprendizagem de Máquina. Pipeline de coleta,
preparação, análise, agrupamento e classificação de dados do mercado de
criptomoedas. **Trabalho educacional — não constitui recomendação de investimento.**

## Estrutura

```
projeto-cripto/
├── data/
│   ├── raw/          # respostas brutas da API (não versionadas)
│   └── processed/    # tabela processada em Parquet
├── notebooks/        # análise exploratória
├── src/
│   ├── collect.py        # coleta (Encontro 3)     ✅ pronto
│   ├── preprocess.py     # limpeza (Encontro 4)
│   ├── features.py       # atributos (Encontro 7)
│   ├── clustering.py     # agrupamento (Encontros 8-9)
│   └── classification.py # classificação (Encontros 10-11)
├── app/              # dashboard/interface
├── reports/          # relatório técnico
├── requirements.txt
└── README.md
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Coleta de dados

Baixa as N maiores moedas por capitalização (CoinGecko) e salva uma tabela longa
`data/processed/precos.parquet` — uma observação por (ativo, data).

```bash
python src/collect.py
```

Chave Demo opcional (100 req/min, 10k chamadas/mês) — sem ela, usa a rota pública:

```bash
export COINGECKO_API_KEY="sua_chave"   # opcional
python src/collect.py
```

Parâmetros no topo de `src/collect.py`: `N_MOEDAS` (30–100), `DIAS` (janela
histórica), `VS_CURRENCY`.

### Tabela gerada (`precos.parquet`)

| coluna | descrição |
|---|---|
| id | identificador CoinGecko (ex.: `bitcoin`) |
| simbolo | símbolo (ex.: `btc`) |
| nome | nome do ativo |
| data | data da observação (UTC) |
| preco | preço de fechamento em `VS_CURRENCY` |
| capitalizacao | capitalização de mercado |
| volume | volume negociado no dia |
| fonte | origem dos dados (`coingecko`) |
| coletado_em | momento da coleta (UTC, ISO 8601) |

## Reprodutibilidade

Os dados brutos ficam em `data/raw/` e a versão processada em `data/processed/`.
Para a avaliação final, recomenda-se congelar uma cópia do Parquet e versioná-la
(ou entregá-la em separado), garantindo reprodução mesmo se a API mudar.
