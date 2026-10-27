"""Testes para src.features.form."""

from src.data.cleaner import limpar_dados
from src.features.form import (
    calcular_forma_recente,
    calcular_media_gols,
    calcular_desempenho_casa_fora,
    calcular_elo,
)


class TestCalcularFormaRecente:

    def test_cria_colunas_esperadas(self, df_processado_exemplo):
        for coluna in (
            "vitorias_recentes_mandante", "empates_recentes_mandante",
            "derrotas_recentes_mandante", "vitorias_recentes_visitante",
            "empates_recentes_visitante", "derrotas_recentes_visitante",
        ):
            assert coluna in df_processado_exemplo.columns

    def test_primeiro_jogo_de_um_time_tem_forma_zerada(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())
        df = calcular_forma_recente(df)

        primeiro_jogo = df.iloc[0]
        assert primeiro_jogo["vitorias_recentes_mandante"] == 0
        assert primeiro_jogo["empates_recentes_mandante"] == 0
        assert primeiro_jogo["derrotas_recentes_mandante"] == 0

    def test_contagens_nao_ultrapassam_a_janela(self, df_processado_exemplo, monkeypatch):
        """Com janela padrão de 5 jogos, vitorias+empates+derrotas <= 5."""
        soma = (
            df_processado_exemplo["vitorias_recentes_mandante"]
            + df_processado_exemplo["empates_recentes_mandante"]
            + df_processado_exemplo["derrotas_recentes_mandante"]
        )
        assert (soma <= 5).all()


class TestCalcularMediaGols:

    def test_cria_colunas_esperadas(self, df_processado_exemplo):
        for coluna in (
            "media_gols_marcados_mandante", "media_gols_sofridos_mandante",
            "media_gols_marcados_visitante", "media_gols_sofridos_visitante",
        ):
            assert coluna in df_processado_exemplo.columns

    def test_medias_nao_sao_negativas(self, df_processado_exemplo):
        for coluna in (
            "media_gols_marcados_mandante", "media_gols_sofridos_mandante",
            "media_gols_marcados_visitante", "media_gols_sofridos_visitante",
        ):
            assert (df_processado_exemplo[coluna] >= 0).all()

    def test_primeiro_jogo_de_um_time_tem_media_zerada(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())
        df = calcular_media_gols(df)

        primeiro_jogo = df.iloc[0]
        assert primeiro_jogo["media_gols_marcados_mandante"] == 0
        assert primeiro_jogo["media_gols_sofridos_mandante"] == 0


class TestCalcularDesempenhoCasaFora:

    def test_cria_colunas_esperadas(self, df_processado_exemplo):
        for coluna in (
            "vitorias_casa", "empates_casa", "derrotas_casa",
            "vitorias_fora", "empates_fora", "derrotas_fora",
        ):
            assert coluna in df_processado_exemplo.columns

    def test_contagens_nao_sao_negativas(self, df_processado_exemplo):
        for coluna in (
            "vitorias_casa", "empates_casa", "derrotas_casa",
            "vitorias_fora", "empates_fora", "derrotas_fora",
        ):
            assert (df_processado_exemplo[coluna] >= 0).all()


class TestCalcularElo:

    def test_cria_colunas_esperadas(self, df_processado_exemplo):
        for coluna in ("elo_mandante_antes", "elo_visitante_antes", "diferenca_elo"):
            assert coluna in df_processado_exemplo.columns

    def test_primeiro_jogo_de_cada_time_comeca_no_elo_inicial(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())
        df = calcular_elo(df, elo_inicial=1500)

        primeiro_jogo = df.iloc[0]
        assert primeiro_jogo["elo_mandante_antes"] == 1500
        assert primeiro_jogo["elo_visitante_antes"] == 1500

    def test_diferenca_elo_bate_com_a_subtracao(self, df_processado_exemplo):
        esperado = (
            df_processado_exemplo["elo_mandante_antes"]
            - df_processado_exemplo["elo_visitante_antes"]
        )
        assert (df_processado_exemplo["diferenca_elo"] == esperado).all()


def test_jogos_no_mesmo_dia_nao_usam_resultados_um_do_outro(df_bruto_exemplo):
    df_bruto_exemplo.loc[1, "data"] = df_bruto_exemplo.loc[0, "data"]
    df_bruto_exemplo.loc[0, "hora"] = "15:00"
    df_bruto_exemplo.loc[1, "hora"] = "20:00"

    df = limpar_dados(df_bruto_exemplo.copy())
    df = calcular_forma_recente(df)
    df = calcular_media_gols(df)
    df = calcular_desempenho_casa_fora(df)
    df = calcular_elo(df)

    segundo_jogo_simultaneo = df.loc[df["ID"] == 2].iloc[0]

    assert segundo_jogo_simultaneo["pontos_mandante_antes"] == 0
    assert segundo_jogo_simultaneo["vitorias_recentes_mandante"] == 0
    assert segundo_jogo_simultaneo["media_gols_marcados_mandante"] == 0
    assert segundo_jogo_simultaneo["vitorias_casa"] == 0
    assert segundo_jogo_simultaneo["elo_mandante_antes"] == 1500