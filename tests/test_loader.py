"""Testes para src.data.loader (item 75: arquivo não encontrado)."""

import pandas as pd
import pytest

from src.data import loader


class TestCarregarDados:

    def test_levanta_filenotfound_quando_csv_nao_existe(self, monkeypatch, tmp_path):
        caminho_falso = tmp_path / "nao_existe.csv"
        monkeypatch.setattr(loader, "CAMINHO_DADOS_RAW", caminho_falso)

        with pytest.raises(FileNotFoundError):
            loader.carregar_dados()

    def test_mensagem_de_erro_menciona_o_caminho_esperado(self, monkeypatch, tmp_path):
        caminho_falso = tmp_path / "nao_existe.csv"
        monkeypatch.setattr(loader, "CAMINHO_DADOS_RAW", caminho_falso)

        with pytest.raises(FileNotFoundError, match="nao_existe.csv"):
            loader.carregar_dados()

    def test_carrega_csv_existente_como_dataframe(self, monkeypatch, tmp_path):
        caminho_csv = tmp_path / "dados.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(caminho_csv, index=False)

        monkeypatch.setattr(loader, "CAMINHO_DADOS_RAW", caminho_csv)

        df = loader.carregar_dados()

        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_normaliza_nomes_de_times_da_fonte_2026(self):
        partidas = pd.DataFrame({
            "data": ["2026-03-01"] * len(loader.ALIASES_TIMES),
            "rodada": ["1ª rodada"] * len(loader.ALIASES_TIMES),
            "mandante": list(loader.ALIASES_TIMES),
            "visitante": ["Adversário"] * len(loader.ALIASES_TIMES),
            "gols_mandante": [1] * len(loader.ALIASES_TIMES),
            "gols_visitante": [0] * len(loader.ALIASES_TIMES),
        })

        resultado = loader._formatar_partidas(partidas)

        assert resultado["mandante"].tolist() == list(
            loader.ALIASES_TIMES.values()
        )