import pandas as pd


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


def calcular_forma_recente(df, ultimos_jogos=5):
    print("\nCalculando forma recente dos times...")

    # Garante que os jogos estejam em ordem cronológica
    df = df.sort_values(["data", "hora", "ID"]).copy()

    # Dicionários para armazenar os últimos resultados
    historico_times = {}

    # Listas que receberão as novas informações
    vitorias_mandante = []
    empates_mandante = []
    derrotas_mandante = []

    vitorias_visitante = []
    empates_visitante = []
    derrotas_visitante = []

    # Percorre cada partida
    for grupo in _grupos_temporais(df):
        for jogo in grupo:
            mandante = jogo.mandante
            visitante = jogo.visitante
            ultimos_m = historico_times.get(mandante, [])[-ultimos_jogos:]
            ultimos_v = historico_times.get(visitante, [])[-ultimos_jogos:]

            vitorias_mandante.append(ultimos_m.count("V"))
            empates_mandante.append(ultimos_m.count("E"))
            derrotas_mandante.append(ultimos_m.count("D"))
            vitorias_visitante.append(ultimos_v.count("V"))
            empates_visitante.append(ultimos_v.count("E"))
            derrotas_visitante.append(ultimos_v.count("D"))

        for jogo in grupo:
            mandante = jogo.mandante
            visitante = jogo.visitante
            if jogo.resultado == "Mandante":
                historico_times.setdefault(mandante, []).append("V")
                historico_times.setdefault(visitante, []).append("D")
            elif jogo.resultado == "Visitante":
                historico_times.setdefault(mandante, []).append("D")
                historico_times.setdefault(visitante, []).append("V")
            else:
                historico_times.setdefault(mandante, []).append("E")
                historico_times.setdefault(visitante, []).append("E")

    # Adiciona as features ao DataFrame
    df["vitorias_recentes_mandante"] = vitorias_mandante
    df["empates_recentes_mandante"] = empates_mandante
    df["derrotas_recentes_mandante"] = derrotas_mandante

    df["vitorias_recentes_visitante"] = vitorias_visitante
    df["empates_recentes_visitante"] = empates_visitante
    df["derrotas_recentes_visitante"] = derrotas_visitante

    print("Forma recente calculada!")

    return df


def calcular_media_gols(df, ultimos_jogos=5):
    
    df = df.sort_values(
        ["data", "hora", "ID"]
    ).reset_index(drop=True)
        
    print("\nCalculando médias de gols recentes...")

    # Dicionário para guardar o histórico de gols de cada time
    historico_gols = {}

    # Listas para armazenar as novas features
    media_gols_marcados_mandante = []
    media_gols_sofridos_mandante = []

    media_gols_marcados_visitante = []
    media_gols_sofridos_visitante = []

    # Percorre as partidas na ordem cronológica
    for grupo in _grupos_temporais(df):
        for jogo in grupo:
            ultimos_m = historico_gols.get(jogo.mandante, [])[-ultimos_jogos:]
            ultimos_v = historico_gols.get(jogo.visitante, [])[-ultimos_jogos:]

            media_gols_marcados_mandante.append(
                sum(gols[0] for gols in ultimos_m) / len(ultimos_m)
                if ultimos_m else 0
            )
            media_gols_sofridos_mandante.append(
                sum(gols[1] for gols in ultimos_m) / len(ultimos_m)
                if ultimos_m else 0
            )
            media_gols_marcados_visitante.append(
                sum(gols[0] for gols in ultimos_v) / len(ultimos_v)
                if ultimos_v else 0
            )
            media_gols_sofridos_visitante.append(
                sum(gols[1] for gols in ultimos_v) / len(ultimos_v)
                if ultimos_v else 0
            )

        for jogo in grupo:
            historico_gols.setdefault(jogo.mandante, []).append(
                (jogo.mandante_Placar, jogo.visitante_Placar)
            )
            historico_gols.setdefault(jogo.visitante, []).append(
                (jogo.visitante_Placar, jogo.mandante_Placar)
            )

    # Adiciona as novas features fora do loop
    df["media_gols_marcados_mandante"] = media_gols_marcados_mandante
    df["media_gols_sofridos_mandante"] = media_gols_sofridos_mandante
    df["media_gols_marcados_visitante"] = media_gols_marcados_visitante
    df["media_gols_sofridos_visitante"] = media_gols_sofridos_visitante

    print("Médias de gols calculadas!")

    return df


def calcular_desempenho_casa_fora(df):
    
    
    df = df.sort_values(
        ["data", "hora", "ID"]
    ).reset_index(drop=True)
    print("\nCalculando desempenho casa/fora...")

    desempenho_casa = {}
    desempenho_fora = {}

    vitorias_casa = []
    empates_casa = []
    derrotas_casa = []

    vitorias_fora = []
    empates_fora = []
    derrotas_fora = []

    for grupo in _grupos_temporais(df):
        for jogo in grupo:
            casa = desempenho_casa.get(jogo.mandante, {"V": 0, "E": 0, "D": 0})
            fora = desempenho_fora.get(jogo.visitante, {"V": 0, "E": 0, "D": 0})
            vitorias_casa.append(casa["V"])
            empates_casa.append(casa["E"])
            derrotas_casa.append(casa["D"])
            vitorias_fora.append(fora["V"])
            empates_fora.append(fora["E"])
            derrotas_fora.append(fora["D"])

        for jogo in grupo:
            casa = desempenho_casa.setdefault(
                jogo.mandante, {"V": 0, "E": 0, "D": 0}
            )
            fora = desempenho_fora.setdefault(
                jogo.visitante, {"V": 0, "E": 0, "D": 0}
            )

            if jogo.resultado == "Mandante":
                casa["V"] += 1
                fora["D"] += 1
            elif jogo.resultado == "Visitante":
                casa["D"] += 1
                fora["V"] += 1
            else:
                casa["E"] += 1
                fora["E"] += 1

    df["vitorias_casa"] = vitorias_casa
    df["empates_casa"] = empates_casa
    df["derrotas_casa"] = derrotas_casa

    df["vitorias_fora"] = vitorias_fora
    df["empates_fora"] = empates_fora
    df["derrotas_fora"] = derrotas_fora

    print("Desempenho casa/fora calculado!")

    return df

def calcular_elo(df, k=20, elo_inicial=1500):
    print("\nCalculando ELO rating e diferença de ELO...")

    elos = {}
    elo_mandante_antes = []
    elo_visitante_antes = []

    # Garante a ordem cronológica
    df = df.sort_values(["data", "hora", "ID"]).copy()

    for grupo in _grupos_temporais(df):
        ajustes_elo = {}

        for jogo in grupo:
            mandante = jogo.mandante
            visitante = jogo.visitante
            elo_m = elos.get(mandante, elo_inicial)
            elo_v = elos.get(visitante, elo_inicial)
            elo_mandante_antes.append(elo_m)
            elo_visitante_antes.append(elo_v)

            exp_m = 1 / (1 + 10 ** ((elo_v - elo_m) / 400))
            exp_v = 1 - exp_m
            if jogo.resultado == "Mandante":
                res_m, res_v = 1.0, 0.0
            elif jogo.resultado == "Visitante":
                res_m, res_v = 0.0, 1.0
            else:
                res_m, res_v = 0.5, 0.5

            ajustes_elo[mandante] = ajustes_elo.get(mandante, 0) + k * (res_m - exp_m)
            ajustes_elo[visitante] = ajustes_elo.get(visitante, 0) + k * (res_v - exp_v)

        for time, ajuste in ajustes_elo.items():
            elos[time] = elos.get(time, elo_inicial) + ajuste

    # 1. Atribui as colunas de ELO antes do jogo
    df["elo_mandante_antes"] = elo_mandante_antes
    df["elo_visitante_antes"] = elo_visitante_antes

    # 2. Cria explicitamente a coluna diferenca_elo
    df["diferenca_elo"] = df["elo_mandante_antes"] - df["elo_visitante_antes"]

    print("Colunas de ELO e diferenca_elo criadas!")
    return df