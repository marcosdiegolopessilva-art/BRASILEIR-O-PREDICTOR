"""Testes para src.models.poisson_model."""

import pytest

from src.models.poisson_model import calcular_probabilidades_poisson


class TestCalcularProbabilidadesPoisson:

    def test_probabilidades_somam_um(self):
        prob_m, prob_e, prob_v = calcular_probabilidades_poisson(1.5, 1.0)

        assert prob_m + prob_e + prob_v == pytest.approx(1.0, abs=1e-6)

    def test_times_com_forca_igual_tem_mandante_igual_visitante(self):
        """
        Sem nenhum ajuste de mando de campo na fórmula (o "bônus
        casa" já está embutido no ataque/defesa calculados antes),
        gols esperados iguais devem dar prob_mandante == prob_visitante.
        """
        prob_m, prob_e, prob_v = calcular_probabilidades_poisson(1.2, 1.2)

        assert prob_m == pytest.approx(prob_v, abs=1e-9)

    def test_mandante_mais_forte_tem_maior_probabilidade(self):
        prob_m, prob_e, prob_v = calcular_probabilidades_poisson(2.5, 0.5)

        assert prob_m > prob_e
        assert prob_m > prob_v

    def test_gols_esperados_zero_favorece_empate_0x0(self):
        prob_m, prob_e, prob_v = calcular_probabilidades_poisson(0.0, 0.0)

        assert prob_e > prob_m
        assert prob_e > prob_v

    def test_todas_as_probabilidades_sao_nao_negativas(self):
        prob_m, prob_e, prob_v = calcular_probabilidades_poisson(3.0, 3.0)

        assert prob_m >= 0
        assert prob_e >= 0
        assert prob_v >= 0