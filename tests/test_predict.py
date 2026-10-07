"""
Testes para src.models.predict.

Cobre o item 74 (previsão) e o item 75 (casos extremos: time
inexistente, poucos jogos / dados insuficientes).
"""

import pandas as pd
import pytest

from src.data.cleaner import limpar_dados
from src.features.form import calcular_elo
from src.models.predict import obter_dados_time, calcular_previsao
from src.models.train_model import treinar_modelo_producao


class TestObterDadosTime:

    def test_time_inexistente_levanta_valueerror(self, df_processado_exemplo):
        with pytest.raises(ValueError):
            obter_dados_time(df_processado_exemplo, "Time Que Não Existe")

    def test_time_existente_retorna_dicionario_com_as_chaves_esperadas(
        self, df_processado_exemplo
    ):
        dados = obter_dados_time(df_processado_exemplo, "Time A")

        chaves_esperadas = {
            "vitorias", "empates", "derrotas",
            "gols_marcados", "gols_sofridos",
            "gols_marcados_casa", "gols_sofridos_casa",
            "gols_marcados_fora", "gols_sofridos_fora",
            "pontos", "vitorias_casa", "empates_casa", "derrotas_casa",
            "vitorias_fora", "empates_fora", "derrotas_fora", "elo",
        }
        assert set(dados) == chaves_esperadas

    def test_pontos_e_vitorias_nao_sao_negativos(self, df_processado_exemplo):
        dados = obter_dados_time(df_processado_exemplo, "Time A")

        assert dados["pontos"] >= 0
        assert dados["vitorias"] >= 0
        assert dados["vitorias_casa"] >= 0


class TestObterDadosTimeComPoucosJogos:
    """Item 75: time com dados insuficientes (nunca jogou em casa)."""

    def test_time_sem_jogos_em_casa_usa_media_geral_como_fallback(self):

        jogos = pd.DataFrame([
            {
                "ID": 1, "rodata": 1, "data": "01/03/2020", "hora": "16:00",
                "mandante": "Time X", "visitante": "Time Estreante",
                "formacao_mandante": "", "formacao_visitante": "",
                "tecnico_mandante": "", "tecnico_visitante": "",
                "vencedor": "Time X", "arena": "Estádio",
                "mandante_Placar": 2, "visitante_Placar": 1,
                "mandante_Estado": "SP", "visitante_Estado": "RJ",
            },
            {
                "ID": 2, "rodata": 2, "data": "08/03/2020", "hora": "16:00",
                "mandante": "Time Y", "visitante": "Time Estreante",
                "formacao_mandante": "", "formacao_visitante": "",
                "tecnico_mandante": "", "tecnico_visitante": "",
                "vencedor": "-", "arena": "Estádio",
                "mandante_Placar": 1, "visitante_Placar": 1,
                "mandante_Estado": "SP", "visitante_Estado": "RJ",
            },
        ])

        df = limpar_dados(jogos)
        df = calcular_elo(df)

        dados = obter_dados_time(df, "Time Estreante")

        # "Time Estreante" só jogou fora -- gols_marcados_casa deve
        # cair no fallback (média geral), não quebrar nem virar NaN.
        assert dados["gols_marcados_casa"] == dados["gols_marcados"]
        assert dados["gols_sofridos_casa"] == dados["gols_sofridos"]


class TestCalcularPrevisao:

    @pytest.fixture
    def modelo_e_scaler(self, df_processado_exemplo):
        resultado = treinar_modelo_producao(df_processado_exemplo)
        return resultado["modelo_logistico"], resultado["scaler"]

    def test_time_mandante_inexistente_levanta_valueerror(
        self, df_processado_exemplo, modelo_e_scaler
    ):
        modelo, scaler = modelo_e_scaler

        with pytest.raises(ValueError):
            calcular_previsao(
                modelo, df_processado_exemplo,
                "Time Fantasma", "Time A", scaler=scaler
            )

    def test_retorna_todas_as_chaves_esperadas(
        self, df_processado_exemplo, modelo_e_scaler
    ):
        modelo, scaler = modelo_e_scaler

        resultado = calcular_previsao(
            modelo, df_processado_exemplo, "Time A", "Time B", scaler=scaler
        )

        chaves_esperadas = {
            "prob_mandante", "prob_empate", "prob_visitante",
            "prob_mandante_poisson", "prob_empate_poisson", "prob_visitante_poisson",
            "prob_mandante_final", "prob_empate_final", "prob_visitante_final",
            "previsao", "confianca",
            "gols_esperados_mandante", "gols_esperados_visitante",
            "placares",
        }
        assert set(resultado) == chaves_esperadas

    def test_probabilidades_finais_somam_um(
        self, df_processado_exemplo, modelo_e_scaler
    ):
        modelo, scaler = modelo_e_scaler

        resultado = calcular_previsao(
            modelo, df_processado_exemplo, "Time A", "Time B", scaler=scaler
        )

        soma = (
            resultado["prob_mandante_final"]
            + resultado["prob_empate_final"]
            + resultado["prob_visitante_final"]
        )
        assert soma == pytest.approx(1.0, abs=1e-6)

    def test_confianca_bate_com_a_maior_probabilidade(
        self, df_processado_exemplo, modelo_e_scaler
    ):
        modelo, scaler = modelo_e_scaler

        resultado = calcular_previsao(
            modelo, df_processado_exemplo, "Time A", "Time B", scaler=scaler
        )

        maior_probabilidade = max(
            resultado["prob_mandante_final"],
            resultado["prob_empate_final"],
            resultado["prob_visitante_final"],
        )
        assert resultado["confianca"] == pytest.approx(maior_probabilidade)