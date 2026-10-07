"""Testes para src.data.cleaner."""

from src.data.cleaner import limpar_dados


class TestLimparDados:

    def test_cria_colunas_essenciais(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())

        for coluna in (
            "resultado", "resultado_encoded", "saldo_gols",
            "ano", "mes", "dia_semana",
            "pontos_mandante_antes", "pontos_visitante_antes",
        ):
            assert coluna in df.columns

    def test_resultado_bate_com_o_placar(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())

        vitorias_mandante = df[df["mandante_Placar"] > df["visitante_Placar"]]
        assert (vitorias_mandante["resultado"] == "Mandante").all()

        vitorias_visitante = df[df["mandante_Placar"] < df["visitante_Placar"]]
        assert (vitorias_visitante["resultado"] == "Visitante").all()

        empates = df[df["mandante_Placar"] == df["visitante_Placar"]]
        assert (empates["resultado"] == "Empate").all()

    def test_resultado_encoded_bate_com_o_mapa(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())

        mapa = {"Empate": 0, "Mandante": 1, "Visitante": 2}
        for resultado, codigo in mapa.items():
            subconjunto = df[df["resultado"] == resultado]
            assert (subconjunto["resultado_encoded"] == codigo).all()

    def test_pontos_resetam_a_cada_temporada(self, df_bruto_exemplo):
        """O primeiro jogo de cada temporada começa com 0 pontos para os dois lados."""
        df = limpar_dados(df_bruto_exemplo.copy())

        for ano in df["ano"].unique():
            primeiro_jogo_do_ano = df[df["ano"] == ano].iloc[0]
            assert primeiro_jogo_do_ano["pontos_mandante_antes"] == 0
            assert primeiro_jogo_do_ano["pontos_visitante_antes"] == 0

    def test_pontos_nunca_sao_negativos(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())

        assert (df["pontos_mandante_antes"] >= 0).all()
        assert (df["pontos_visitante_antes"] >= 0).all()

    def test_nao_ha_valores_ausentes(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())

        assert df.isnull().sum().sum() == 0

    def test_nao_ha_linhas_duplicadas(self, df_bruto_exemplo):
        df = limpar_dados(df_bruto_exemplo.copy())

        assert df.duplicated().sum() == 0