import pandas as pd

from src.config import CAMINHO_DADOS_PROCESSED


def _grupos_temporais(df):
    linhas = list(df.itertuples(index=False))
    datas = df["data"].to_numpy()
    inicio = 0

    for indice in range(1, len(linhas) + 1):
        mesmo_grupo = (
            indice < len(linhas)
            and (
                datas[indice] == datas[inicio]
                or (pd.isna(datas[indice]) and pd.isna(datas[inicio]))
            )
        )
        if mesmo_grupo:
            continue

        yield linhas[inicio:indice]
        inicio = indice


def limpar_dados(df):

    print("\nIniciando limpeza da base...")

    # Converter a coluna de data para datetime
    df["data"] = pd.to_datetime(
        df["data"],
        dayfirst=True
    )

    # Remover espaços extras dos nomes dos times
    df["mandante"] = df["mandante"].str.strip()
    df["visitante"] = df["visitante"].str.strip()

    # Remover linhas duplicadas
    df = df.drop_duplicates()

    # Remover colunas que não serão utilizadas
    colunas_remover = [
        "formacao_mandante",
        "formacao_visitante",
        "tecnico_mandante",
        "tecnico_visitante"
    ]

    df = df.drop(columns=colunas_remover)

    # Corrigir nome da coluna rodada
    df = df.rename(
        columns={
            "rodata": "rodada"
        }
    )

    print("Limpeza concluída!")

    print("\nCriando novas variáveis...")

    # Resultado da partida
    df["resultado"] = "Empate"

    df.loc[
        df["mandante_Placar"] > df["visitante_Placar"],
        "resultado"
    ] = "Mandante"

    df.loc[
        df["mandante_Placar"] < df["visitante_Placar"],
        "resultado"
    ] = "Visitante"

    mapa_resultado = {
        "Empate": 0,
        "Mandante": 1,
        "Visitante": 2
    }

    df["resultado_encoded"] = (
        df["resultado"].map(mapa_resultado)
    )

    df["saldo_gols"] = (
        df["mandante_Placar"] - df["visitante_Placar"]
    )

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia_semana"] = df["data"].dt.dayofweek

    print("\nCalculando histórico de pontos...")

    df = df.sort_values(
        ["data", "hora", "ID"]
    ).reset_index(drop=True)

    pontos_times = {}
    pontos_mandante = []
    pontos_visitante = []
    ano_atual = None

    for grupo in _grupos_temporais(df):
        ano_jogo = grupo[0].ano
        if ano_jogo != ano_atual:
            pontos_times = {}
            ano_atual = ano_jogo

        for jogo in grupo:
            pontos_mandante.append(
                pontos_times.get(jogo.mandante, 0)
            )
            pontos_visitante.append(
                pontos_times.get(jogo.visitante, 0)
            )

        for jogo in grupo:
            mandante = jogo.mandante
            visitante = jogo.visitante

            if jogo.resultado == "Mandante":
                pontos_times[mandante] = (
                    pontos_times.get(mandante, 0) + 3
                )
            elif jogo.resultado == "Visitante":
                pontos_times[visitante] = (
                    pontos_times.get(visitante, 0) + 3
                )
            else:
                pontos_times[mandante] = (
                    pontos_times.get(mandante, 0) + 1
                )
                pontos_times[visitante] = (
                    pontos_times.get(visitante, 0) + 1
                )

    df["pontos_mandante_antes"] = pontos_mandante
    df["pontos_visitante_antes"] = pontos_visitante

    print("Histórico de pontos calculado!")

    caminho_saida = CAMINHO_DADOS_PROCESSED

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        caminho_saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Base salva em: {caminho_saida}")

    return df