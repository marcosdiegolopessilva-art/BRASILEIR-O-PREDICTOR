import pandas as pd


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
    for jogo in df.itertuples(index=False):
        mandante = jogo.mandante
        visitante = jogo.visitante

        # Histórico anterior do mandante
        historico_m = historico_times.get(mandante, [])

        # Histórico anterior do visitante
        historico_v = historico_times.get(visitante, [])

        # Considera somente os últimos jogos
        ultimos_m = historico_m[-ultimos_jogos:]
        ultimos_v = historico_v[-ultimos_jogos:]

        # Calcula a forma do mandante
        vitorias_mandante.append(ultimos_m.count("V"))
        empates_mandante.append(ultimos_m.count("E"))
        derrotas_mandante.append(ultimos_m.count("D"))

        # Calcula a forma do visitante
        vitorias_visitante.append(ultimos_v.count("V"))
        empates_visitante.append(ultimos_v.count("E"))
        derrotas_visitante.append(ultimos_v.count("D"))

        # Atualiza o histórico depois da partida
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
    for jogo in df.itertuples(index=False):
        mandante = jogo.mandante
        visitante = jogo.visitante

        # Histórico anterior do mandante
        historico_m = historico_gols.get(mandante, [])

        # Histórico anterior do visitante
        historico_v = historico_gols.get(visitante, [])

        # Considera somente os últimos jogos
        ultimos_m = historico_m[-ultimos_jogos:]
        ultimos_v = historico_v[-ultimos_jogos:]

        # Média de gols do mandante
        if ultimos_m:
            media_marcados_m = sum(j[0] for j in ultimos_m) / len(ultimos_m)
            media_sofridos_m = sum(j[1] for j in ultimos_m) / len(ultimos_m)
        else:
            media_marcados_m = 0
            media_sofridos_m = 0

        # Média de gols do visitante
        if ultimos_v:
            media_marcados_v = sum(j[0] for j in ultimos_v) / len(ultimos_v)
            media_sofridos_v = sum(j[1] for j in ultimos_v) / len(ultimos_v)
        else:
            media_marcados_v = 0
            media_sofridos_v = 0

        # Guarda os resultados
        media_gols_marcados_mandante.append(media_marcados_m)
        media_gols_sofridos_mandante.append(media_sofridos_m)

        media_gols_marcados_visitante.append(media_marcados_v)
        media_gols_sofridos_visitante.append(media_sofridos_v)

        # Atualiza o histórico depois da partida
        historico_gols.setdefault(mandante, []).append(
            (jogo.mandante_Placar, jogo.visitante_Placar)
        )
        historico_gols.setdefault(visitante, []).append(
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

    for jogo in df.itertuples(index=False):
        mandante = jogo.mandante
        visitante = jogo.visitante

        casa = desempenho_casa.get(
            mandante,
            {"V": 0, "E": 0, "D": 0}
        )

        fora = desempenho_fora.get(
            visitante,
            {"V": 0, "E": 0, "D": 0}
        )

        # Guarda histórico antes da partida
        vitorias_casa.append(casa["V"])
        empates_casa.append(casa["E"])
        derrotas_casa.append(casa["D"])

        vitorias_fora.append(fora["V"])
        empates_fora.append(fora["E"])
        derrotas_fora.append(fora["D"])

        # Atualiza após o jogo
        if jogo.resultado == "Mandante":
            desempenho_casa.setdefault(
                mandante,
                {"V": 0, "E": 0, "D": 0}
            )["V"] += 1

            desempenho_fora.setdefault(
                visitante,
                {"V": 0, "E": 0, "D": 0}
            )["D"] += 1

        elif jogo.resultado == "Visitante":
            desempenho_casa.setdefault(
                mandante,
                {"V": 0, "E": 0, "D": 0}
            )["D"] += 1

            desempenho_fora.setdefault(
                visitante,
                {"V": 0, "E": 0, "D": 0}
            )["V"] += 1

        else:
            desempenho_casa.setdefault(
                mandante,
                {"V": 0, "E": 0, "D": 0}
            )["E"] += 1

            desempenho_fora.setdefault(
                visitante,
                {"V": 0, "E": 0, "D": 0}
            )["E"] += 1

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

    for jogo in df.itertuples(index=False):
        mandante = jogo.mandante
        visitante = jogo.visitante

        # ELO dos times ANTES da partida
        elo_m = elos.get(mandante, elo_inicial)
        elo_v = elos.get(visitante, elo_inicial)

        elo_mandante_antes.append(elo_m)
        elo_visitante_antes.append(elo_v)

        # Probabilidade esperada
        exp_m = 1 / (1 + 10 ** ((elo_v - elo_m) / 400))
        exp_v = 1 - exp_m

        # Resultado real do confronto
        if jogo.resultado == "Mandante":
            res_m, res_v = 1.0, 0.0
        elif jogo.resultado == "Visitante":
            res_m, res_v = 0.0, 1.0
        else:
            res_m, res_v = 0.5, 0.5

        # Atualiza o histórico para o próximo jogo
        elos[mandante] = elo_m + k * (res_m - exp_m)
        elos[visitante] = elo_v + k * (res_v - exp_v)

    # 1. Atribui as colunas de ELO antes do jogo
    df["elo_mandante_antes"] = elo_mandante_antes
    df["elo_visitante_antes"] = elo_visitante_antes

    # 2. Cria explicitamente a coluna diferenca_elo
    df["diferenca_elo"] = df["elo_mandante_antes"] - df["elo_visitante_antes"]

    print("Colunas de ELO e diferenca_elo criadas!")
    return df