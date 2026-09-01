"""
collect.py — Coleta de dados históricos da Solana (CoinGecko).

O projeto usa **apenas um ativo** (por padrão `solana`). O script baixa a
série histórica diária (preço, capitalização e volume) desse ativo e a
salva numa tabela longa.

Saídas:
  data/raw/moedas.json           -> metadados do(s) ativo(s) coletado(s)
  data/raw/<id>.json             -> resposta bruta da API, por ativo
  data/processed/precos.parquet  -> tabela longa: uma observação por (ativo, data)

Formato da tabela final (uma observação por ativo-data):
  id, simbolo, nome, data, preco, capitalizacao, volume, fonte, coletado_em

Encontro 3 do cronograma (Coleta e armazenamento).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
BASE_URL = "https://api.coingecko.com/api/v3"
VS_CURRENCY = "usd"          # moeda de referência
DIAS = 365                   # janela histórica (>90 => granularidade diária automática)

# Ativo único do projeto. Configurável pelo ambiente, mas o padrão é a Solana.
# Ex.:  export COINGECKO_COIN_ID="solana"
MOEDA_ID = os.environ.get("COINGECKO_COIN_ID", "solana").strip() or "solana"

# Chave Demo opcional. Sem chave usa a rota pública (limite menor).
# Defina no terminal antes de rodar:  export COINGECKO_API_KEY="sua_chave"
API_KEY = os.environ.get("COINGECKO_API_KEY", "").strip()

# Pausa entre requisições (Demo: 100 req/min). 0.8s ~ 75 req/min, folga segura.
PAUSA_S = 0.8
MAX_TENTATIVAS = 4           # re-tentativas em caso de 429/erro de rede

# Caminhos (independentes de onde o script é chamado)
RAIZ = Path(__file__).resolve().parents[1]   # .../projeto-cripto
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"
SNAPSHOT = RAIZ / "data" / "snapshot" / "solana_365d.csv"

# Snapshot congelado e versionado no git. A API devolve uma janela **móvel** de
# DIAS dias, então rodar a coleta em outra data produz outra série e os números
# dos relatórios deixam de bater. Para reproduzir exatamente os resultados
# entregues, rode com o snapshot:
#     USAR_SNAPSHOT=1 python src/collect.py
# Sem a variável, o script busca dados novos na API (o diferencial de
# "atualização automática" do projeto).
USAR_SNAPSHOT = os.environ.get("USAR_SNAPSHOT", "").strip().lower() in {
    "1", "true", "sim", "s",
}


# ----------------------------------------------------------------------
# Utilidades de requisição
# ----------------------------------------------------------------------
def _headers() -> dict:
    return {"x-cg-demo-api-key": API_KEY} if API_KEY else {}


def _get(endpoint: str, params: dict):
    """GET com re-tentativa e backoff simples para o limite de taxa (429)."""
    url = f"{BASE_URL}{endpoint}"
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        resp = requests.get(url, params=params, headers=_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:            # excedeu o limite -> espera e repete
            espera = 5 * tentativa
            print(f"  [429] limite atingido, aguardando {espera}s...")
            time.sleep(espera)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Falha ao consultar {endpoint} após {MAX_TENTATIVAS} tentativas")


# ----------------------------------------------------------------------
# 1. Metadados do ativo único
# ----------------------------------------------------------------------
def obter_meta(coin_id: str) -> dict:
    """Retorna id, símbolo, nome e rank do ativo pedido (via /coins/markets)."""
    dados = _get(
        "/coins/markets",
        {
            "vs_currency": VS_CURRENCY,
            "ids": coin_id,
            "sparkline": "false",
        },
    )
    if not dados:
        raise RuntimeError(f"Ativo '{coin_id}' não encontrado na CoinGecko")
    m = dados[0]
    return {
        "id": m["id"],
        "simbolo": m["symbol"],
        "nome": m["name"],
        "rank": m.get("market_cap_rank"),
    }


# ----------------------------------------------------------------------
# 2. Série histórica de um ativo
# ----------------------------------------------------------------------
def baixar_historico(coin_id: str) -> dict:
    """Baixa preço, capitalização e volume diários dos últimos DIAS dias."""
    return _get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": VS_CURRENCY, "days": DIAS},
        # sem 'interval': com days>90 a API já devolve granularidade diária
    )


def serie_para_df(bruto: dict, meta: dict) -> pd.DataFrame:
    """Converte a resposta bruta em linhas (ativo, data, preço, cap, volume)."""
    precos = pd.DataFrame(bruto["prices"], columns=["ts", "preco"])
    caps = pd.DataFrame(bruto["market_caps"], columns=["ts", "capitalizacao"])
    vols = pd.DataFrame(bruto["total_volumes"], columns=["ts", "volume"])

    df = precos.merge(caps, on="ts").merge(vols, on="ts")
    df["data"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
    # mantém uma linha por dia (descarta o ponto parcial do dia corrente)
    df = df.drop_duplicates(subset="data", keep="first")

    df.insert(0, "id", meta["id"])
    df.insert(1, "simbolo", meta["simbolo"])
    df.insert(2, "nome", meta["nome"])
    return df[["id", "simbolo", "nome", "data", "preco", "capitalizacao", "volume"]]


# ----------------------------------------------------------------------
# 3. Orquestração
# ----------------------------------------------------------------------
def carregar_snapshot() -> pd.DataFrame:
    """Lê a série congelada e versionada, sem tocar na API."""
    if not SNAPSHOT.exists():
        raise SystemExit(f"Snapshot não encontrado: {SNAPSHOT}")
    dados = pd.read_csv(SNAPSHOT, parse_dates=["data"])
    dados["data"] = dados["data"].dt.date
    return dados.sort_values(["id", "data"]).reset_index(drop=True)


def main() -> None:
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    destino = DIR_PROC / "precos.parquet"

    if USAR_SNAPSHOT:
        print(f"Usando snapshot congelado: {SNAPSHOT.relative_to(RAIZ)}")
        dados = carregar_snapshot()
        dados.to_parquet(destino, index=False)
        print(
            f"\nOK: {len(dados)} linhas, {dados['id'].nunique()} ativo(s), "
            f"de {dados['data'].min()} a {dados['data'].max()}"
        )
        print(f"Arquivo salvo em: {destino}")
        return

    momento = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"Coletando apenas o ativo '{MOEDA_ID}'...")
    meta = obter_meta(MOEDA_ID)
    (DIR_RAW / "moedas.json").write_text(
        json.dumps([meta], ensure_ascii=False, indent=2)
    )

    cid = meta["id"]
    print(f"Baixando série histórica de {cid} ({DIAS} dias)...")
    bruto = baixar_historico(cid)
    (DIR_RAW / f"{cid}.json").write_text(json.dumps(bruto))
    time.sleep(PAUSA_S)

    dados = serie_para_df(bruto, meta)
    dados["fonte"] = "coingecko"
    dados["coletado_em"] = momento
    dados = dados.sort_values(["id", "data"]).reset_index(drop=True)
    dados.to_parquet(destino, index=False)

    print(
        f"\nOK: {len(dados)} linhas, {dados['id'].nunique()} ativo(s), "
        f"de {dados['data'].min()} a {dados['data'].max()}"
    )
    print(f"Arquivo salvo em: {destino}")


if __name__ == "__main__":
    main()
