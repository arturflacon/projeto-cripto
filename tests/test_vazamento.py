"""test_vazamento.py — Verificações automáticas de vazamento temporal.

O argumento central deste projeto é metodológico: **os atributos usam apenas
dados até o instante t; só o rótulo olha para o futuro, e só no treino.** Uma
afirmação dessas não deve ficar apenas no texto do relatório — este arquivo a
transforma em teste executável.

Rodar (da raiz do projeto):
    python tests/test_vazamento.py

Também funciona sob pytest, se estiver instalado:
    pytest tests/ -v

Cada função ``teste_*`` levanta ``AssertionError`` se a propriedade for violada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import classification            # noqa: E402
import features                  # noqa: E402
import regression                # noqa: E402

PRECOS = RAIZ / "data" / "processed" / "precos_limpo.parquet"


def _carregar() -> pd.DataFrame:
    if not PRECOS.exists():
        raise SystemExit(
            f"Arquivo não encontrado: {PRECOS}\n"
            "Rode antes: python src/collect.py && python src/preprocess.py"
        )
    return pd.read_parquet(PRECOS)


# ----------------------------------------------------------------------
# 1. O teste central: truncar a série não pode mudar o passado
# ----------------------------------------------------------------------
def teste_atributos_nao_veem_o_futuro() -> str:
    """Recomputar os atributos sobre uma série truncada deve dar o mesmo valor.

    Se qualquer atributo usasse informação posterior a `t` (uma janela centrada,
    uma normalização com estatística global, um `shift` negativo), remover o
    final da série mudaria os valores das datas anteriores ao corte. Este é o
    teste que fecha a questão para os 13 atributos de uma vez.
    """
    precos = _carregar()
    colunas = features.colunas_atributos()

    corte = precos["data"].quantile(0.7)
    truncado = precos[precos["data"] <= corte]

    completo = features.construir_atributos(precos).set_index("data")
    parcial = features.construir_atributos(truncado).set_index("data")

    comuns = parcial.index
    cheio = completo.loc[comuns, colunas]
    cortado = parcial[colunas]

    diferenca = (cheio - cortado).abs().max().max()

    # Comparar só os valores deixa escapar o vazamento mais óbvio: um `shift(-1)`
    # é localmente consistente e só difere na ÚLTIMA linha do trecho truncado,
    # onde vira NaN — e `.max()` ignora NaN silenciosamente. Por isso o padrão de
    # NaN também precisa bater. Todo atributo aqui olha só para trás, então
    # nenhum deles pode ser NaN no fim do trecho e ter valor na série completa.
    nan_divergentes = int((cheio.isna() != cortado.isna()).to_numpy().sum())

    assert nan_divergentes == 0, (
        f"VAZAMENTO: {nan_divergentes} posições viraram NaN ao truncar a série — "
        "algum atributo depende de um dia posterior (típico de `shift` negativo)"
    )
    assert diferenca == 0.0, (
        f"VAZAMENTO: atributos mudaram ao truncar a série "
        f"(maior diferença {diferenca:.3e}); algum atributo olha para o futuro"
    )
    return (
        f"corte em {corte.date()}, diferença máxima {diferenca:.1e} e "
        f"{nan_divergentes} divergências de NaN nos {len(colunas)} atributos"
    )


# ----------------------------------------------------------------------
# 2. O rótulo olha para o futuro — e para o futuro certo
# ----------------------------------------------------------------------
def teste_rotulo_corresponde_ao_dia_seguinte() -> str:
    """``y=1`` deve equivaler exatamente a ``preco(t+1) > preco(t)``."""
    precos = _carregar()
    tabela = features.montar_tabela(precos, "direcao")
    serie = precos.set_index("data")["preco"]

    esperado = [
        int(serie[d + pd.Timedelta(days=1)] > serie[d]) for d in tabela["data"]
    ]
    divergentes = int((np.array(esperado) != tabela["y"].to_numpy()).sum())

    assert divergentes == 0, f"{divergentes} rótulos não batem com o preço de t+1"
    return f"{len(tabela)} rótulos conferidos contra os preços reais"


def teste_ultima_linha_descartada() -> str:
    """O último dia não tem "amanhã": deve sair da tabela de treino/avaliação."""
    precos = _carregar()
    tabela = features.montar_tabela(precos, "direcao")
    ultimo_dado = precos["data"].max()

    assert tabela["data"].max() < ultimo_dado, (
        "a última linha (sem rótulo possível) não foi descartada"
    )
    # e com manter_sem_rotulo=True ela volta, sem rótulo — é a linha do prever.py
    com_pendente = features.montar_tabela(precos, "direcao", manter_sem_rotulo=True)
    pendentes = com_pendente[com_pendente["y"].isna()]

    assert len(pendentes) >= 1, "manter_sem_rotulo=True não preservou o dia pendente"
    assert pendentes["data"].max() == ultimo_dado, "o dia pendente não é o último dia"
    return f"último dado {ultimo_dado.date()} fora do treino, disponível para previsão"


def teste_mediana_do_alvo_alternativo_nao_usa_o_futuro() -> str:
    """No alvo ``acima_mediana``, o limiar deve vir só dos retornos passados.

    Usar a mediana do período inteiro seria vazamento sutil: o limiar carregaria
    informação de dias que ainda não aconteceram.
    """
    precos = _carregar()
    corte = precos["data"].quantile(0.7)
    truncado = precos[precos["data"] <= corte]

    completo = features.construir_atributos(precos)
    parcial = features.construir_atributos(truncado)
    alvo_completo = features.construir_alvo(completo, "acima_mediana")
    alvo_parcial = features.construir_alvo(parcial, "acima_mediana")

    # compara só onde ambos têm rótulo definido (o último dia do truncado não tem)
    n = len(parcial) - 1
    a = alvo_completo.iloc[:n].astype("float")
    b = alvo_parcial.iloc[:n].astype("float")
    divergentes = int((a.fillna(-1) != b.fillna(-1)).sum())

    assert divergentes == 0, (
        f"VAZAMENTO: {divergentes} rótulos 'acima_mediana' mudaram ao truncar a "
        "série — o limiar está usando dados futuros"
    )
    return f"{n} rótulos estáveis sob truncamento"


# ----------------------------------------------------------------------
# 3. Alvo da regressão
# ----------------------------------------------------------------------
def teste_alvo_de_volatilidade() -> str:
    """A volatilidade futura deve ser o desvio dos retornos de t+1 até t+H."""
    precos = _carregar()
    horizonte = 7
    com_atributos = features.construir_atributos(precos)
    alvo = regression.volatilidade_futura(com_atributos["ret_1d"], horizonte)

    for posicao in (100, 150, 200):
        manual = com_atributos["ret_1d"].iloc[posicao + 1: posicao + 1 + horizonte].std()
        assert np.isclose(alvo.iloc[posicao], manual), (
            f"alvo divergente na posição {posicao}: "
            f"{alvo.iloc[posicao]:.6f} != {manual:.6f}"
        )

    assert alvo.tail(horizonte).isna().all(), (
        "as últimas linhas deveriam ser NaN (janela futura incompleta)"
    )
    return f"3 posições conferidas na mão; últimas {horizonte} linhas são NaN"


# ----------------------------------------------------------------------
# 4. O split é temporal, não aleatório
# ----------------------------------------------------------------------
def teste_split_e_cronologico() -> str:
    """Todo dia de treino deve vir antes de todo dia de teste, sem interseção."""
    precos = _carregar()
    tabela = features.montar_tabela(precos, "direcao")
    treino, teste = classification.dividir_temporal(tabela)

    assert treino["data"].max() < teste["data"].min(), (
        "há dias de treino posteriores ao início do teste"
    )
    intersecao = set(treino["data"]) & set(teste["data"])
    assert not intersecao, f"{len(intersecao)} datas aparecem nos dois conjuntos"

    return (
        f"treino até {treino['data'].max().date()}, "
        f"teste a partir de {teste['data'].min().date()}"
    )


def teste_escala_ajustada_so_no_treino() -> str:
    """O ``StandardScaler`` do Pipeline deve ver apenas o treino.

    Se a padronização fosse ajustada na tabela inteira, média e desvio
    carregariam informação do teste — vazamento clássico e fácil de cometer.
    """
    precos = _carregar()
    tabela = features.montar_tabela(precos, "direcao")
    treino, _ = classification.dividir_temporal(tabela)
    colunas = features.colunas_atributos()

    modelo = classification.construir_modelos()["KNN"]
    modelo.fit(treino[colunas], treino["y"])
    escala = modelo.named_steps["escala"]

    media_treino = treino[colunas].mean().to_numpy()
    media_total = tabela[colunas].mean().to_numpy()

    assert np.allclose(escala.mean_, media_treino), (
        "a escala não corresponde às estatísticas do treino"
    )
    assert not np.allclose(escala.mean_, media_total), (
        "VAZAMENTO: a escala foi ajustada com a tabela inteira, incluindo o teste"
    )
    return "média do scaler == média do treino, e diferente da média total"


# ----------------------------------------------------------------------
# 5. Integridade da tabela
# ----------------------------------------------------------------------
def teste_tabela_sem_nan_nem_infinito() -> str:
    """A tabela de atributos entregue ao modelo não pode ter NaN nem infinitos."""
    precos = _carregar()
    tabela = features.montar_tabela(precos, "direcao")
    colunas = features.colunas_atributos()

    nans = int(tabela[colunas].isna().sum().sum())
    infinitos = int(np.isinf(tabela[colunas].to_numpy()).sum())

    assert nans == 0, f"{nans} valores NaN na tabela de atributos"
    assert infinitos == 0, f"{infinitos} valores infinitos na tabela de atributos"
    return f"{len(tabela)} linhas x {len(colunas)} atributos, sem NaN nem infinitos"


def teste_persistencia_usa_so_o_rotulo_de_ontem() -> str:
    """``y_anterior`` (baseline de persistência) deve ser o rótulo de t-1, fora de X.

    O rótulo de t-1 depende de ``preco(t)`` e ``preco(t-1)`` — já conhecidos em t.
    No alvo ``direcao`` ele coincide com ``ret_1d > 0``; nos dois alvos deve ser
    igual ao ``y`` do dia anterior.
    """
    precos = _carregar()
    assert "y_anterior" not in features.colunas_atributos(), "y_anterior não pode estar em X"

    for alvo in ("direcao", "acima_mediana"):
        tabela = features.montar_tabela(precos, alvo).set_index("data")
        datas_ontem = tabela.index - pd.Timedelta(days=1)
        tem_ontem = datas_ontem.isin(tabela.index)
        esperado = tabela["y"].reindex(datas_ontem[tem_ontem]).to_numpy()
        obtido = tabela.loc[tem_ontem, "y_anterior"].astype(int).to_numpy()
        assert (esperado == obtido).all(), f"[{alvo}] y_anterior != y do dia anterior"

    direcao = features.montar_tabela(precos, "direcao")
    assert (direcao["y_anterior"].astype(int) == (direcao["ret_1d"] > 0).astype(int)).all(), (
        "no alvo direcao, a persistência deveria coincidir com ret_1d > 0"
    )
    return "y_anterior = rótulo de t-1 nos dois alvos, fora de X"


# ----------------------------------------------------------------------
# Execução direta
# ----------------------------------------------------------------------
TESTES = [
    teste_atributos_nao_veem_o_futuro,
    teste_rotulo_corresponde_ao_dia_seguinte,
    teste_ultima_linha_descartada,
    teste_mediana_do_alvo_alternativo_nao_usa_o_futuro,
    teste_alvo_de_volatilidade,
    teste_split_e_cronologico,
    teste_escala_ajustada_so_no_treino,
    teste_tabela_sem_nan_nem_infinito,
    teste_persistencia_usa_so_o_rotulo_de_ontem,
]


def main() -> int:
    """Roda todos os testes e devolve o código de saída (0 = tudo passou)."""
    print("=" * 72)
    print("  VERIFICAÇÕES DE VAZAMENTO TEMPORAL")
    print("=" * 72)

    falhas = 0
    for teste in TESTES:
        nome = teste.__name__.replace("teste_", "").replace("_", " ")
        try:
            detalhe = teste()
            print(f"\n  [OK]    {nome}")
            print(f"          {detalhe}")
        except AssertionError as erro:
            falhas += 1
            print(f"\n  [FALHA] {nome}")
            print(f"          {erro}")

    print("\n" + "=" * 72)
    if falhas:
        print(f"  {falhas} de {len(TESTES)} verificações FALHARAM")
    else:
        print(f"  todas as {len(TESTES)} verificações passaram")
    print("=" * 72)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
