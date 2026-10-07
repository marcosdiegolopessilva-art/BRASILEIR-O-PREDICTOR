import pandas as pd
from src.models.monte_carlo import simular_placares_monte_carlo
from src.models.poisson_model import calcular_probabilidades_poisson
from src.models.train_model import FEATURES
from src.config import ULTIMOS_JOGOS_FORMA, N_SIMULACOES_MONTE_CARLO


def obter_dados_time(df, time, ano=None, ultimos_jogos=ULTIMOS_JOGOS_FORMA):

    df_completo = df.sort_values(
        ["data", "hora", "ID"]
    ).reset_index(drop=True)

    temporada_referencia = (
        ano if ano is not None else df_completo["ano"].max()
    )

    if ano is not None:
        df_historico = df_completo[df_completo["ano"] == ano].copy()
    else:
        df_historico = df_completo

    jogos_casa = df_historico[
        df_historico["mandante"] == time
    ].copy()

    jogos_fora = df_historico[
        df_historico["visitante"] == time
    ].copy()

    jogos = pd.concat(
        [jogos_casa, jogos_fora]
    ).sort_values(
        ["data", "hora", "ID"]
    )

    if jogos.empty:
        raise ValueError(
            f"Time '{time}' não encontrado na base de dados "
            "(nenhuma partida registrada). Confira a grafia do "
            "nome exatamente como aparece na coluna 'mandante' "
            "ou 'visitante' do CSV."
        )

    ultimos = jogos.tail(ultimos_jogos)

    vitorias = 0
    empates = 0
    derrotas = 0

    gols_marcados = 0
    gols_sofridos = 0

    for _, jogo in ultimos.iterrows():

        if jogo["mandante"] == time:

            gols_marcados += jogo["mandante_Placar"]
            gols_sofridos += jogo["visitante_Placar"]

            if jogo["resultado"] == "Mandante":
                vitorias += 1
            elif jogo["resultado"] == "Empate":
                empates += 1
            else:
                derrotas += 1

        else:

            gols_marcados += jogo["visitante_Placar"]
            gols_sofridos += jogo["mandante_Placar"]

            if jogo["resultado"] == "Visitante":
                vitorias += 1
            elif jogo["resultado"] == "Empate":
                empates += 1
            else:
                derrotas += 1

    jogos_temporada = jogos[jogos["ano"] == temporada_referencia]

    pontos = 0

    for _, jogo in jogos_temporada.iterrows():

        if jogo["mandante"] == time:

            if jogo["resultado"] == "Mandante":
                pontos += 3
            elif jogo["resultado"] == "Empate":
                pontos += 1

        elif jogo["visitante"] == time:

            if jogo["resultado"] == "Visitante":
                pontos += 3
            elif jogo["resultado"] == "Empate":
                pontos += 1

    vitorias_casa = 0
    empates_casa = 0
    derrotas_casa = 0

    for _, jogo in jogos_casa.iterrows():

        if jogo["resultado"] == "Mandante":
            vitorias_casa += 1
        elif jogo["resultado"] == "Empate":
            empates_casa += 1
        else:
            derrotas_casa += 1

    vitorias_fora = 0
    empates_fora = 0
    derrotas_fora = 0

    for _, jogo in jogos_fora.iterrows():

        if jogo["resultado"] == "Visitante":
            vitorias_fora += 1
        elif jogo["resultado"] == "Empate":
            empates_fora += 1
        else:
            derrotas_fora += 1

    ultimo_jogo = jogos.iloc[-1]

    if ultimo_jogo["mandante"] == time:

        resultado_pontual = (
            1.0 if ultimo_jogo["resultado"] == "Mandante"
            else 0.5 if ultimo_jogo["resultado"] == "Empate"
            else 0.0
        )

        esperado = 1 / (
            1 + 10 ** (
                (ultimo_jogo["elo_visitante_antes"] - ultimo_jogo["elo_mandante_antes"])
                / 400
            )
        )

        elo = ultimo_jogo["elo_mandante_antes"] + 20 * (resultado_pontual - esperado)

    else:

        resultado_pontual = (
            1.0 if ultimo_jogo["resultado"] == "Visitante"
            else 0.5 if ultimo_jogo["resultado"] == "Empate"
            else 0.0
        )

        esperado = 1 / (
            1 + 10 ** (
                (ultimo_jogo["elo_mandante_antes"] - ultimo_jogo["elo_visitante_antes"])
                / 400
            )
        )

        elo = ultimo_jogo["elo_visitante_antes"] + 20 * (resultado_pontual - esperado)

    media_geral_marcados = gols_marcados / len(ultimos)
    media_geral_sofridos = gols_sofridos / len(ultimos)

    if not jogos_casa.empty:
        gols_marcados_casa = jogos_casa["mandante_Placar"].mean()
        gols_sofridos_casa = jogos_casa["visitante_Placar"].mean()
    else:
        gols_marcados_casa = media_geral_marcados
        gols_sofridos_casa = media_geral_sofridos

    if not jogos_fora.empty:
        gols_marcados_fora = jogos_fora["visitante_Placar"].mean()
        gols_sofridos_fora = jogos_fora["mandante_Placar"].mean()
    else:
        gols_marcados_fora = media_geral_marcados
        gols_sofridos_fora = media_geral_sofridos

    return {

        "vitorias": vitorias,
        "empates": empates,
        "derrotas": derrotas,

        "gols_marcados": media_geral_marcados,
        "gols_sofridos": media_geral_sofridos,

        "gols_marcados_casa": gols_marcados_casa,
        "gols_sofridos_casa": gols_sofridos_casa,
        "gols_marcados_fora": gols_marcados_fora,
        "gols_sofridos_fora": gols_sofridos_fora,

        "pontos": pontos,

        "vitorias_casa": vitorias_casa,
        "empates_casa": empates_casa,
        "derrotas_casa": derrotas_casa,

        "vitorias_fora": vitorias_fora,
        "empates_fora": empates_fora,
        "derrotas_fora": derrotas_fora,

        "elo": elo
    }


def calcular_previsao(modelo, df, mandante, visitante, ano=None, scaler=None):
    """
    Calcula a previsão completa de uma partida -- probabilidades
    da Regressão Logística, do Poisson, o Ensemble final, gols
    esperados e os placares mais prováveis -- SEM imprimir nada.

    Separada de prever_partida() para que a mesma lógica sirva
    tanto para o console (main.py, via prever_partida) quanto
    para a interface Streamlit (app.py), sem duplicar código.

    Parâmetros
    ----------
    modelo : sklearn.linear_model.LogisticRegression
        Modelo treinado (ex.: resultado_producao["modelo_logistico"]).
    df : pandas.DataFrame
        Base processada (mesma usada no treino).
    mandante, visitante : str
        Nomes dos times, como aparecem na base.
    ano : int ou None
        Ver obter_dados_time().
    scaler : sklearn.preprocessing.StandardScaler ou None
        Necessário para a Regressão Logística.

    Retorna
    -------
    dict
        prob_mandante, prob_empate, prob_visitante (Regressão Logística)
        prob_mandante_poisson, prob_empate_poisson, prob_visitante_poisson
        prob_mandante_final, prob_empate_final, prob_visitante_final
        previsao (str), confianca (float)
        gols_esperados_mandante, gols_esperados_visitante
        placares (lista de tuplas ((gols_m, gols_v), quantidade))

    Levanta
    -------
    ValueError
        Se `mandante` ou `visitante` não existirem na base
        (propagado de obter_dados_time()).
    """

    dados_mandante = obter_dados_time(df, mandante, ano)
    dados_visitante = obter_dados_time(df, visitante, ano)

    entrada = pd.DataFrame([{

        "pontos_mandante_antes": dados_mandante["pontos"],
        "pontos_visitante_antes": dados_visitante["pontos"],

        "vitorias_recentes_mandante": dados_mandante["vitorias"],
        "empates_recentes_mandante": dados_mandante["empates"],
        "derrotas_recentes_mandante": dados_mandante["derrotas"],

        "vitorias_recentes_visitante": dados_visitante["vitorias"],
        "empates_recentes_visitante": dados_visitante["empates"],
        "derrotas_recentes_visitante": dados_visitante["derrotas"],

        "media_gols_marcados_mandante": dados_mandante["gols_marcados"],
        "media_gols_sofridos_mandante": dados_mandante["gols_sofridos"],

        "media_gols_marcados_visitante": dados_visitante["gols_marcados"],
        "media_gols_sofridos_visitante": dados_visitante["gols_sofridos"],

        "vitorias_casa": dados_mandante["vitorias_casa"],
        "empates_casa": dados_mandante["empates_casa"],
        "derrotas_casa": dados_mandante["derrotas_casa"],

        "vitorias_fora": dados_visitante["vitorias_fora"],
        "empates_fora": dados_visitante["empates_fora"],
        "derrotas_fora": dados_visitante["derrotas_fora"],

        "dif_pontos": dados_mandante["pontos"] - dados_visitante["pontos"],
        "dif_vitorias_recentes": dados_mandante["vitorias"] - dados_visitante["vitorias"],
        "dif_empates_recentes": dados_mandante["empates"] - dados_visitante["empates"],
        "dif_derrotas_recentes": dados_mandante["derrotas"] - dados_visitante["derrotas"],
        "dif_gols_marcados": dados_mandante["gols_marcados"] - dados_visitante["gols_marcados"],
        "dif_gols_sofridos": dados_mandante["gols_sofridos"] - dados_visitante["gols_sofridos"],
        "dif_vitorias_casa_fora": dados_mandante["vitorias_casa"] - dados_visitante["vitorias_fora"],
        "dif_empates_casa_fora": dados_mandante["empates_casa"] - dados_visitante["empates_fora"],
        "dif_derrotas_casa_fora": dados_mandante["derrotas_casa"] - dados_visitante["derrotas_fora"],

        "diferenca_elo": dados_mandante["elo"] - dados_visitante["elo"]

    }])

    entrada = entrada[FEATURES]

    if scaler is not None:
        entrada_para_modelo = scaler.transform(entrada)
    else:
        entrada_para_modelo = entrada

    probabilidades = modelo.predict_proba(entrada_para_modelo)[0]
    classes = modelo.classes_

    probabilidades_por_classe = dict(zip(classes, probabilidades))

    prob_empate = probabilidades_por_classe.get(0, 0)
    prob_mandante = probabilidades_por_classe.get(1, 0)
    prob_visitante = probabilidades_por_classe.get(2, 0)

    # Poisson (ataque/defesa por mando de campo -- item 47) e
    # Ensemble (Fase 13): validado no Walk-Forward como o
    # resultado final -- ver src/models/hybrid.py.
    ataque_mandante = dados_mandante["gols_marcados_casa"]
    defesa_mandante = dados_mandante["gols_sofridos_casa"]
    ataque_visitante = dados_visitante["gols_marcados_fora"]
    defesa_visitante = dados_visitante["gols_sofridos_fora"]

    gols_esperados_mandante = (ataque_mandante + defesa_visitante) / 2
    gols_esperados_visitante = (ataque_visitante + defesa_mandante) / 2

    prob_mandante_poisson, prob_empate_poisson, prob_visitante_poisson = (
        calcular_probabilidades_poisson(
            gols_esperados_mandante, gols_esperados_visitante
        )
    )

    prob_mandante_final = (prob_mandante + prob_mandante_poisson) / 2
    prob_empate_final = (prob_empate + prob_empate_poisson) / 2
    prob_visitante_final = (prob_visitante + prob_visitante_poisson) / 2

    probabilidades_final = {
        0: prob_empate_final, 1: prob_mandante_final, 2: prob_visitante_final
    }
    classe_predita = max(probabilidades_final, key=probabilidades_final.get)

    if classe_predita == 0:
        previsao = "EMPATE"
    elif classe_predita == 1:
        previsao = f"VITÓRIA DO {mandante.upper()}"
    else:
        previsao = f"VITÓRIA DO {visitante.upper()}"

    confianca = probabilidades_final[classe_predita]

    simulacao_placares = simular_placares_monte_carlo(
        gols_esperados_mandante, gols_esperados_visitante,
        simulacoes=N_SIMULACOES_MONTE_CARLO
    )

    return {
        "prob_mandante": prob_mandante,
        "prob_empate": prob_empate,
        "prob_visitante": prob_visitante,

        "prob_mandante_poisson": prob_mandante_poisson,
        "prob_empate_poisson": prob_empate_poisson,
        "prob_visitante_poisson": prob_visitante_poisson,

        "prob_mandante_final": prob_mandante_final,
        "prob_empate_final": prob_empate_final,
        "prob_visitante_final": prob_visitante_final,

        "previsao": previsao,
        "confianca": confianca,

        "gols_esperados_mandante": gols_esperados_mandante,
        "gols_esperados_visitante": gols_esperados_visitante,

        "placares": simulacao_placares["placares"],
    }


def prever_partida(modelo, df, mandante, visitante, ano=None, scaler=None, nome_modelo="Modelo"):
    """
    Versão de console: calcula a previsão (via calcular_previsao)
    e imprime tudo formatado. Usada por main.py.

    Parâmetros
    ----------
    scaler : sklearn.preprocessing.StandardScaler ou None
        Passe o scaler retornado por treinar_modelo() quando
        `modelo` for a Regressão Logística.
    nome_modelo : str
        Mantido por compatibilidade; não é mais usado nos prints.
    """

    print("\n" + "=" * 50)
    print("PREVISÃO DA PARTIDA")
    print("=" * 50)

    print(f"\nMandante: {mandante}")
    print(f"Visitante: {visitante}")

    resultado = calcular_previsao(modelo, df, mandante, visitante, ano, scaler)

    print("\n" + "=" * 50)
    print("PROBABILIDADES POR MODELO")
    print("=" * 50)

    print(f"\nRegressão Logística:")
    print(f"  Vitória do {mandante}: {resultado['prob_mandante']:.2%}")
    print(f"  Empate: {resultado['prob_empate']:.2%}")
    print(f"  Vitória do {visitante}: {resultado['prob_visitante']:.2%}")

    print(f"\nPoisson:")
    print(f"  Vitória do {mandante}: {resultado['prob_mandante_poisson']:.2%}")
    print(f"  Empate: {resultado['prob_empate_poisson']:.2%}")
    print(f"  Vitória do {visitante}: {resultado['prob_visitante_poisson']:.2%}")

    print("\n" + "=" * 50)
    print("PROBABILIDADE FINAL (ENSEMBLE)")
    print("=" * 50)

    print(f"\nVitória do {mandante}: {resultado['prob_mandante_final']:.2%}")
    print(f"Empate: {resultado['prob_empate_final']:.2%}")
    print(f"Vitória do {visitante}: {resultado['prob_visitante_final']:.2%}")

    print(f"\nPREVISÃO: {resultado['previsao']}")
    print(f"CONFIANÇA: {resultado['confianca']:.2%}")

    print("\n" + "=" * 50)
    print("GOLS ESPERADOS")
    print("=" * 50)
    print(f"\n{mandante}: {resultado['gols_esperados_mandante']:.2f}")
    print(f"{visitante}: {resultado['gols_esperados_visitante']:.2f}")

    print("\n" + "=" * 50)
    print("PLACARES MAIS PROVÁVEIS")
    print("=" * 50)
    print(f"\n{N_SIMULACOES_MONTE_CARLO:,} simulações".replace(",", "."))

    for placar, quantidade in resultado["placares"]:
        percentual = quantidade / N_SIMULACOES_MONTE_CARLO
        print(f"{placar[0]} x {placar[1]} → {percentual:.2%}")

    return {
        0: resultado["prob_empate_final"],
        1: resultado["prob_mandante_final"],
        2: resultado["prob_visitante_final"],
    }