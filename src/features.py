"""features.py — Engenharia de atributos (Encontro 7).

Transforma a série diária limpa da Solana numa **tabela supervisionada**:
cada linha é um dia `t`; os atributos usam somente dados até `t` e o rótulo
olha para `t+1`.

Entrada:  ``data/processed/precos_limpo.parquet``
Saída:    ``data/processed/atributos.parquet``

Regra central (sem vazamento temporal): toda janela móvel termina em `t`
(``rolling``/``expanding`` incluem o dia corrente e o passado, nunca o futuro).
O único olhar para o futuro é o rótulo ``y``, e ele só é usado no treino.

Trabalho acadêmico: nada aqui é recomendação de investimento.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuração (parâmetros registrados para reprodutibilidade)
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]          # .../projeto-cripto
ENTRADA = RAIZ / "data" / "processed" / "precos_limpo.parquet"
SAIDA = RAIZ / "data" / "processed" / "atributos.parquet"

# Definição do alvo:
#   "direcao"        -> preco(t+1) > preco(t)
#   "acima_mediana"  -> retorno de t+1 acima da mediana histórica dos retornos
ALVO = "direcao"

JANELA_CURTA = 7        # média móvel / volatilidade curta
JANELA_LONGA = 30       # média móvel / volatilidade longa, assimetria e curtose
JANELA_ACUM = 90        # horizonte do retorno acumulado (momento mais longo)
MIN_OBS_MEDIANA = 60    # mínimo de retornos passados para a mediana expandida

# As médias móveis em **nível de preço** (mm_7, mm_30) são calculadas e salvas
# para a EDA, mas ficam fora de X por padrão: são não estacionárias e no teste
# assumem valores fora da faixa vista no treino, o que degrada modelos baseados
# em distância (KNN/SVM). A informação delas entra normalizada em `dist_mm_*`.
INCLUIR_MM_NIVEL = False

# Atributos sempre estacionários / normalizados.
COLUNAS_BASE = [
    "ret_1d", "ret_7d", "ret_30d", "ret_acum",
    "vol_7d", "vol_30d",
    "dist_mm_7", "dist_mm_30", "cruz_mm",
    "drawdown", "razao_vol_cap",
    "assimetria_30", "curtose_30",
]
COLUNAS_MM_NIVEL = ["mm_7", "mm_30"]


def colunas_atributos() -> list[str]:
    """Lista das colunas que compõem X (respeita ``INCLUIR_MM_NIVEL``)."""
    return COLUNAS_BASE + (COLUNAS_MM_NIVEL if INCLUIR_MM_NIVEL else [])


# ----------------------------------------------------------------------
# 1. Atributos (somente passado até t)
# ----------------------------------------------------------------------
def _retorno(preco: pd.Series, k: int) -> pd.Series:
    """Retorno simples de k dias: preco(t)/preco(t-k) - 1.

    Calculado explicitamente (em vez de ``pct_change``) para não preencher
    silenciosamente valores ausentes — lacuna continua lacuna.
    """
    return preco / preco.shift(k) - 1.0


def construir_atributos(df_ativo: pd.DataFrame) -> pd.DataFrame:
    """Constrói o vetor de atributos diário de um ativo.

    Recebe a série já ordenada por data, com índice diário contínuo.
    """
    df = df_ativo.sort_values("data").reset_index(drop=True).copy()
    preco = df["preco"]

    # -- retornos (momento em diferentes horizontes)
    df["ret_1d"] = _retorno(preco, 1)
    df["ret_7d"] = _retorno(preco, JANELA_CURTA)
    df["ret_30d"] = _retorno(preco, JANELA_LONGA)
    df["ret_acum"] = _retorno(preco, JANELA_ACUM)

    # -- volatilidade histórica (desvio-padrão dos retornos diários na janela)
    df["vol_7d"] = df["ret_1d"].rolling(JANELA_CURTA).std()
    df["vol_30d"] = df["ret_1d"].rolling(JANELA_LONGA).std()

    # -- médias móveis e distância percentual do preço à média
    df["mm_7"] = preco.rolling(JANELA_CURTA).mean()
    df["mm_30"] = preco.rolling(JANELA_LONGA).mean()
    df["dist_mm_7"] = preco / df["mm_7"] - 1.0
    df["dist_mm_30"] = preco / df["mm_30"] - 1.0
    # cruzamento de médias: +1 curta acima da longa, -1 abaixo, 0 empate
    df["cruz_mm"] = np.sign(df["mm_7"] - df["mm_30"])

    # -- drawdown: queda percentual em relação ao pico anterior (máximo expandido)
    df["drawdown"] = preco / preco.cummax() - 1.0

    # -- liquidez relativa
    df["razao_vol_cap"] = df["volume"] / df["capitalizacao"]

    # -- forma da distribuição dos retornos na janela longa (opcionais)
    df["assimetria_30"] = df["ret_1d"].rolling(JANELA_LONGA).skew()
    df["curtose_30"] = df["ret_1d"].rolling(JANELA_LONGA).kurt()

    return df


# ----------------------------------------------------------------------
# 2. Rótulo (único olhar para o futuro)
# ----------------------------------------------------------------------
def construir_alvo(df_ativo: pd.DataFrame, alvo: str = ALVO) -> pd.Series:
    """Rótulo binário do dia seguinte.

    * ``"direcao"``: 1 se ``preco(t+1) > preco(t)``.
    * ``"acima_mediana"``: 1 se o retorno de ``t -> t+1`` supera a **mediana
      dos retornos passados até t** (mediana expandida, não a mediana do
      período inteiro — usar o período inteiro vazaria o futuro no limiar).

    Onde não há como decidir (falta o preço de amanhã, ou a mediana ainda não
    tem observações suficientes) o rótulo fica ``NaN`` e a linha é descartada.
    """
    preco = df_ativo["preco"]
    preco_amanha = preco.shift(-1)
    conhecido = preco.notna() & preco_amanha.notna()

    if alvo == "direcao":
        rotulo = preco_amanha > preco
        return rotulo.where(conhecido)

    if alvo == "acima_mediana":
        ret_1d = preco / preco.shift(1) - 1.0
        ret_futuro = preco_amanha / preco - 1.0
        # mediana dos retornos observados até t (inclui t, exclui t+1)
        mediana_passada = ret_1d.expanding(min_periods=MIN_OBS_MEDIANA).median()
        rotulo = ret_futuro > mediana_passada
        return rotulo.where(conhecido & mediana_passada.notna())

    raise ValueError(f"ALVO desconhecido: {alvo!r} (use 'direcao' ou 'acima_mediana')")


# ----------------------------------------------------------------------
# 3. Montagem da tabela supervisionada
# ----------------------------------------------------------------------
def montar_tabela(
    df: pd.DataFrame, alvo: str = ALVO, *, manter_sem_rotulo: bool = False
) -> pd.DataFrame:
    """Monta X + y por ativo e descarta as linhas sem janela completa.

    Por padrão descarta também as linhas sem rótulo — incluindo o **último dia**,
    que ainda não tem "amanhã" para servir de gabarito. Esse é o comportamento
    correto para treinar e avaliar.

    Com ``manter_sem_rotulo=True`` o último dia é preservado (com ``y`` nulo):
    é justamente a linha usada por ``prever.py`` para classificar o dia seguinte,
    que por definição ainda não aconteceu.
    """
    partes: list[pd.DataFrame] = []

    for _, grupo in df.groupby("id", sort=True):
        com_atributos = construir_atributos(grupo)
        com_atributos["y"] = construir_alvo(com_atributos, alvo)
        partes.append(com_atributos)

    tabela = pd.concat(partes, ignore_index=True)

    colunas = ["id", "data", "preco", *colunas_atributos(), "y"]
    tabela = tabela[colunas]

    # o aquecimento das janelas (NaN iniciais) sai sempre
    antes = len(tabela)
    obrigatorias = list(colunas_atributos())
    if not manter_sem_rotulo:
        obrigatorias.append("y")
    tabela = tabela.dropna(subset=obrigatorias).reset_index(drop=True)

    # tipo inteiro; nulável quando as linhas sem rótulo são preservadas
    tabela["y"] = tabela["y"].astype("Int64" if manter_sem_rotulo else "int64")

    print(f"  linhas descartadas (aquecimento{'' if manter_sem_rotulo else ' + última'}): "
          f"{antes - len(tabela)}")
    return tabela.sort_values(["id", "data"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------
def main() -> None:
    if not ENTRADA.exists():
        raise SystemExit(
            f"Arquivo de entrada não encontrado: {ENTRADA}\n"
            "Rode primeiro `python src/preprocess.py`."
        )
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    print(f"Alvo: {ALVO} | janelas: {JANELA_CURTA}/{JANELA_LONGA}/{JANELA_ACUM}")
    bruto = pd.read_parquet(ENTRADA)
    tabela = montar_tabela(bruto, ALVO)
    tabela.to_parquet(SAIDA, index=False)

    proporcao = tabela["y"].mean()
    print(
        f"\nOK: {len(tabela)} linhas x {len(colunas_atributos())} atributos, "
        f"de {tabela['data'].min().date()} a {tabela['data'].max().date()}"
    )
    print(
        f"Balanceamento de y: {proporcao:.1%} da classe 1 "
        f"({int(tabela['y'].sum())} de {len(tabela)})"
    )
    print(f"Arquivo salvo em: {SAIDA}")


if __name__ == "__main__":
    main()
