"""Testes para src.models.metrics (Brier Score, comparação e calibração)."""

import numpy as np
import pytest

from src.models.metrics import (
    calcular_brier_multiclasse,
    comparar_brier,
    curva_calibracao,
)


class TestCalcularBrierMulticlasse:

    def test_previsao_perfeita_da_brier_zero(self):
        y_real = [1, 1, 0, 2]
        probabilidades = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])

        brier = calcular_brier_multiclasse(y_real, probabilidades)

        assert brier == pytest.approx(0.0)

    def test_previsao_totalmente_errada_da_brier_dois(self):
        y_real = [1]
        probabilidades = np.array([[0, 0, 1]])  # certeza na classe errada

        brier = calcular_brier_multiclasse(y_real, probabilidades)

        assert brier == pytest.approx(2.0)

    def test_chute_uniforme_da_brier_conhecido(self):
        y_real = [0]
        probabilidades = np.array([[1 / 3, 1 / 3, 1 / 3]])

        brier = calcular_brier_multiclasse(y_real, probabilidades)

        esperado = (2 / 3) ** 2 + (1 / 3) ** 2 + (1 / 3) ** 2
        assert brier == pytest.approx(esperado)

    def test_formato_incompativel_levanta_valueerror(self):
        y_real = [0, 1, 2]
        probabilidades = np.array([[0.5, 0.5]])  # faltam linhas/colunas

        with pytest.raises(ValueError):
            calcular_brier_multiclasse(y_real, probabilidades)


class TestCompararBrier:

    def test_ordena_do_melhor_para_o_pior(self):
        y_real = [1, 1, 0, 2]

        probs_boas = np.array([
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ])

        probs_ruins = np.array([
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
        ])

        tabela = comparar_brier(y_real, {"Bom": probs_boas, "Ruim": probs_ruins})

        assert list(tabela["modelo"]) == ["Bom", "Ruim"]
        assert tabela.loc[0, "brier_score"] < tabela.loc[1, "brier_score"]


class TestCurvaCalibracao:

    def test_classe_invalida_levanta_valueerror(self):
        with pytest.raises(ValueError):
            curva_calibracao(
                [0, 1], np.array([[0.5, 0.5, 0], [0.5, 0.5, 0]]), classe=5
            )

    def test_probabilidade_constante_gera_uma_faixa_so(self):
        y_real = [1, 0, 1, 1, 0]
        probabilidades = np.array([[0.2, 0.55, 0.25]] * 5)

        tabela = curva_calibracao(y_real, probabilidades, classe=1)

        assert len(tabela) == 1
        assert tabela.loc[0, "quantidade_amostras"] == 5

    def test_faixas_sem_amostra_sao_omitidas(self):
        y_real = [1, 1]
        probabilidades = np.array([[0.1, 0.9, 0.0], [0.1, 0.9, 0.0]])

        tabela = curva_calibracao(y_real, probabilidades, classe=1, n_bins=10)

        assert len(tabela) == 1