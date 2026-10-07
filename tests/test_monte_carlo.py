"""Testes unitários para src.models.monte_carlo.

Estes testes não alteram o código de produção. Eles verificam o contrato
atual das duas funções principais do módulo usando dados sintéticos e
controlando a aleatoriedade quando isso é necessário.
"""

import numpy as np
import pytest

from src.models.monte_carlo import (
    simular_monte_carlo,
    simular_placares_monte_carlo,
)


class TestSimularMonteCarlo:
    """Testes da simulação de resultado da partida."""

    def test_retorna_as_tres_probabilidades(self):
        resultado = simular_monte_carlo(
            prob_mandante=0.50,
            prob_empate=0.25,
            prob_visitante=0.25,
            simulacoes=100,
        )

        assert set(resultado) == {"mandante", "empate", "visitante"}

    def test_probabilidades_ficam_entre_zero_e_um(self):
        resultado = simular_monte_carlo(
            prob_mandante=0.50,
            prob_empate=0.25,
            prob_visitante=0.25,
            simulacoes=1000,
        )

        for probabilidade in resultado.values():
            assert 0 <= probabilidade <= 1

    def test_probabilidades_simuladas_somam_um(self):
        resultado = simular_monte_carlo(
            prob_mandante=0.50,
            prob_empate=0.25,
            prob_visitante=0.25,
            simulacoes=1000,
        )

        assert sum(resultado.values()) == pytest.approx(1.0)

    def test_normaliza_probabilidades_antes_da_simulacao(self, monkeypatch):
        """Verifica que entradas cuja soma não é 1 são normalizadas."""
        chamadas = {}

        def fake_choice(opcoes, size, p):
            chamadas["opcoes"] = opcoes
            chamadas["size"] = size
            chamadas["p"] = p
            return np.array([1, 0, 2, 1])

        monkeypatch.setattr(np.random, "choice", fake_choice)

        resultado = simular_monte_carlo(
            prob_mandante=1.0,
            prob_empate=0.5,
            prob_visitante=0.5,
            simulacoes=4,
        )

        assert chamadas["opcoes"] == [0, 1, 2]
        assert chamadas["size"] == 4
        assert chamadas["p"] == pytest.approx([0.25, 0.50, 0.25])
        assert resultado == {
            "mandante": 0.50,
            "empate": 0.25,
            "visitante": 0.25,
        }

    def test_probabilidade_100_por_cento_produz_um_unico_resultado(self):
        resultado = simular_monte_carlo(
            prob_mandante=1.0,
            prob_empate=0.0,
            prob_visitante=0.0,
            simulacoes=100,
        )

        assert resultado == {
            "mandante": 1.0,
            "empate": 0.0,
            "visitante": 0.0,
        }


class TestSimularPlacaresMonteCarlo:
    """Testes da simulação de placares via Poisson."""

    def test_retorna_todas_as_chaves_esperadas(self):
        resultado = simular_placares_monte_carlo(
            gols_mandante=1.5,
            gols_visitante=1.0,
            simulacoes=100,
        )

        assert set(resultado) == {
            "prob_mandante",
            "prob_empate",
            "prob_visitante",
            "placares",
            "gols_esperados_mandante",
            "gols_esperados_visitante",
        }

    def test_probabilidades_de_placar_somam_um(self):
        resultado = simular_placares_monte_carlo(
            gols_mandante=1.5,
            gols_visitante=1.0,
            simulacoes=1000,
        )

        soma = (
            resultado["prob_mandante"]
            + resultado["prob_empate"]
            + resultado["prob_visitante"]
        )

        assert soma == pytest.approx(1.0)

    def test_probabilidades_de_placar_ficam_entre_zero_e_um(self):
        resultado = simular_placares_monte_carlo(
            gols_mandante=1.5,
            gols_visitante=1.0,
            simulacoes=1000,
        )

        for chave in (
            "prob_mandante",
            "prob_empate",
            "prob_visitante",
        ):
            assert 0 <= resultado[chave] <= 1

    def test_dez_placares_mais_frequentes_no_maximo(self):
        resultado = simular_placares_monte_carlo(
            gols_mandante=1.5,
            gols_visitante=1.0,
            simulacoes=1000,
        )

        assert len(resultado["placares"]) <= 10

        for placar, quantidade in resultado["placares"]:
            assert isinstance(placar, tuple)
            assert len(placar) == 2
            assert quantidade > 0

    def test_gols_negativos_sao_limitados_a_001(self, monkeypatch):
        """Documenta o comportamento atual de proteção da função."""
        lambdas = []

        def fake_poisson(lam, size):
            lambdas.append(lam)
            if len(lambdas) == 1:
                return np.array([1, 0, 1, 0])
            return np.array([0, 0, 1, 0])

        monkeypatch.setattr(np.random, "poisson", fake_poisson)

        resultado = simular_placares_monte_carlo(
            gols_mandante=-5,
            gols_visitante=-2,
            simulacoes=4,
        )

        assert lambdas == [0.01, 0.01]
        assert resultado["gols_esperados_mandante"] == 0.01
        assert resultado["gols_esperados_visitante"] == 0.01

    def test_gols_esperados_validos_sao_preservados(self, monkeypatch):
        lambdas = []

        def fake_poisson(lam, size):
            lambdas.append(lam)
            return np.zeros(size, dtype=int)

        monkeypatch.setattr(np.random, "poisson", fake_poisson)

        resultado = simular_placares_monte_carlo(
            gols_mandante=1.8,
            gols_visitante=0.9,
            simulacoes=4,
        )

        assert lambdas == [1.8, 0.9]
        assert resultado["gols_esperados_mandante"] == 1.8
        assert resultado["gols_esperados_visitante"] == 0.9
