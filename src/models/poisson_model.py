"""
Cálculo exato das probabilidades de resultado (Mandante/Empate/
Visitante) a partir de gols esperados, usando a distribuição de
Poisson -- o mesmo princípio de monte_carlo.py, mas via soma
exata em vez de simulação. Usado para comparar o modelo Poisson
com a Regressão Logística de forma justa (sem ruído de simulação
entrando na comparação de Brier Score).
"""

from scipy.stats import poisson

from src.config import MAX_GOLS_POISSON


def calcular_probabilidades_poisson(
    gols_esperados_mandante, gols_esperados_visitante, max_gols=MAX_GOLS_POISSON
):
    """
    Calcula P(vitória do mandante), P(empate) e P(vitória do
    visitante) somando, para cada combinação de placar até
    `max_gols` gols por time, a probabilidade conjunta
    P(gols_mandante) * P(gols_visitante) -- Poisson independente
    para cada time, a mesma premissa usada em monte_carlo.py.

    Parâmetros
    ----------
    gols_esperados_mandante : float
        Média (lambda) de gols do mandante.
    gols_esperados_visitante : float
        Média (lambda) de gols do visitante.
    max_gols : int
        Placar máximo considerado por time na soma (10 cobre
        praticamente 100% da massa de probabilidade em futebol).

    Retorna
    -------
    tuple[float, float, float]
        (prob_mandante, prob_empate, prob_visitante), somando
        muito próximo de 1.0 (a massa acima de max_gols é
        desprezível).
    """

    prob_mandante = 0.0
    prob_empate = 0.0
    prob_visitante = 0.0

    for gols_m in range(max_gols + 1):
        p_m = poisson.pmf(gols_m, gols_esperados_mandante)

        for gols_v in range(max_gols + 1):
            p_v = poisson.pmf(gols_v, gols_esperados_visitante)
            p_conjunta = p_m * p_v

            if gols_m > gols_v:
                prob_mandante += p_conjunta
            elif gols_m == gols_v:
                prob_empate += p_conjunta
            else:
                prob_visitante += p_conjunta

    return prob_mandante, prob_empate, prob_visitante
