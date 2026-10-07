import numpy as np
from collections import Counter


def simular_monte_carlo(
    prob_mandante,
    prob_empate,
    prob_visitante,
    simulacoes=10000
):
    """
    Simula vitória, empate ou derrota
    usando as probabilidades do modelo.
    """

    probabilidades = np.array([
        prob_empate,
        prob_mandante,
        prob_visitante
    ])

    # Garante que a soma seja 1
    probabilidades = (
        probabilidades
        / probabilidades.sum()
    )

    resultados = np.random.choice(
        [0, 1, 2],
        size=simulacoes,
        p=probabilidades
    )

    quantidade_empates = np.sum(
        resultados == 0
    )

    quantidade_mandante = np.sum(
        resultados == 1
    )

    quantidade_visitante = np.sum(
        resultados == 2
    )

    return {
        "mandante":
            quantidade_mandante / simulacoes,

        "empate":
            quantidade_empates / simulacoes,

        "visitante":
            quantidade_visitante / simulacoes
    }


def simular_placares_monte_carlo(
    gols_mandante,
    gols_visitante,
    simulacoes=10000
):
    """
    Simula placares utilizando
    distribuição de Poisson.
    """

    gols_mandante = max(
        float(gols_mandante),
        0.01
    )

    gols_visitante = max(
        float(gols_visitante),
        0.01
    )

    placares_mandante = np.random.poisson(
        gols_mandante,
        simulacoes
    )

    placares_visitante = np.random.poisson(
        gols_visitante,
        simulacoes
    )

    vitorias_mandante = np.sum(
        placares_mandante > placares_visitante
    )

    empates = np.sum(
        placares_mandante == placares_visitante
    )

    vitorias_visitante = np.sum(
        placares_mandante < placares_visitante
    )

    prob_mandante = (
        vitorias_mandante / simulacoes
    )

    prob_empate = (
        empates / simulacoes
    )

    prob_visitante = (
        vitorias_visitante / simulacoes
    )

    contagem_placares = Counter(
        zip(
            placares_mandante,
            placares_visitante
        )
    )

    placares_mais_frequentes = (
        contagem_placares
        .most_common(10)
    )

    return {
        "prob_mandante":
            prob_mandante,

        "prob_empate":
            prob_empate,

        "prob_visitante":
            prob_visitante,

        "placares":
            placares_mais_frequentes,

        "gols_esperados_mandante":
            gols_mandante,

        "gols_esperados_visitante":
            gols_visitante
    }