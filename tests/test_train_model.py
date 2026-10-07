"""Testes para src.models.train_model."""

import pytest

from src.models.train_model import construir_features, FEATURES, treinar_modelo_producao


class TestConstruirFeatures:

    def test_cria_todas_as_variaveis_de_diferenca(self, df_processado_exemplo):
        df = construir_features(df_processado_exemplo)

        for coluna in FEATURES:
            assert coluna in df.columns

    def test_dif_pontos_bate_com_a_subtracao(self, df_processado_exemplo):
        df = construir_features(df_processado_exemplo)

        esperado = df["pontos_mandante_antes"] - df["pontos_visitante_antes"]
        assert (df["dif_pontos"] == esperado).all()

    def test_dif_vitorias_casa_fora_bate_com_a_subtracao(self, df_processado_exemplo):
        df = construir_features(df_processado_exemplo)

        esperado = df["vitorias_casa"] - df["vitorias_fora"]
        assert (df["dif_vitorias_casa_fora"] == esperado).all()

    def test_nao_altera_o_dataframe_original(self, df_processado_exemplo):
        """construir_features() deve trabalhar em uma cópia, não no df recebido."""
        colunas_antes = set(df_processado_exemplo.columns)

        construir_features(df_processado_exemplo)

        assert set(df_processado_exemplo.columns) == colunas_antes


class TestTreinarModeloProducao:

    def test_treina_com_100_por_cento_dos_dados(self, df_processado_exemplo):
        resultado = treinar_modelo_producao(df_processado_exemplo)

        assert set(resultado) == {"modelo_logistico", "scaler", "features"}
        assert resultado["features"] == FEATURES

    def test_modelo_treinado_faz_previsoes_validas(self, df_processado_exemplo):
        resultado = treinar_modelo_producao(df_processado_exemplo)

        df_com_features = construir_features(df_processado_exemplo)
        entrada = df_com_features[FEATURES].iloc[:1]
        entrada_escalada = resultado["scaler"].transform(entrada)

        probabilidades = resultado["modelo_logistico"].predict_proba(entrada_escalada)[0]

        assert len(probabilidades) == 3
        assert probabilidades.sum() == pytest.approx(1.0)
        assert all(p >= 0 for p in probabilidades)