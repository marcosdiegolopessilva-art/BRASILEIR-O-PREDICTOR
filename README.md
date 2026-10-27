# ⚽ Brasileirão Predictor

Sistema em Python para estimar probabilidades de resultados de partidas do Campeonato Brasileiro. O projeto compara classificadores, avalia probabilidades com métricas próprias para previsões probabilísticas e combina Regressão Logística com Poisson na previsão final.

O sistema estima as probabilidades de:

- Vitória do mandante
- Empate
- Vitória do visitante

E complementa com gols esperados e os placares mais prováveis, via simulação de Monte Carlo.

---

## 🎯 Objetivo

Estimar probabilidades de vitória do mandante, empate e vitória do visitante, avaliando o desempenho com Brier Score, Log Loss, calibração e divisões temporais. Essas métricas ajudam a medir qualidade, mas não garantem que previsões futuras sejam confiáveis.

---

## 📊 Dados

- **8.785 partidas históricas** no CSV original, em 22 temporadas de 2003 a 2024.
- A fonte pública de 2026 retornou 380 jogos; na execução registrada abaixo, 277 tinham placar final e foram acrescentados pelo SQLite, totalizando **9.062 partidas** usadas no treino.
- Colunas originais: data, mandante, visitante, placar, arena, estado dos times
- Distribuição de resultados:

| Resultado | Partidas | Percentual |
|---|---:|---:|
| Vitória do mandante | 4.489 | 49,54% |
| Empate | 2.398 | 26,46% |
| Vitória do visitante | 2.175 | 24,00% |

Essas contagens correspondem à execução com 9.062 partidas; a distribuição muda quando a base é atualizada.

---

## 🧠 Engenharia de Atributos

As features de forma, gols, pontos, casa/fora e ELO são calculadas a partir do histórico anterior. Jogos com a mesma data e horário são processados em grupo: os históricos só são atualizados depois de calcular as features de todos os jogos desse grupo.

**Histórico de pontos** — pontos acumulados na temporada corrente, resetados a cada ano.

**Forma recente** — vitórias, empates e derrotas nos últimos 5 jogos de cada time.

**Médias de gols** — gols marcados e sofridos recentes.

**Desempenho casa/fora** — vitórias, empates e derrotas separadas por mando de campo, histórico completo.

**Ataque/defesa por mando de campo** — média de gols marcados e sofridos *especificamente* em jogos em casa (para o mandante) e fora (para o visitante), usada na estimativa de gols esperados do modelo Poisson.

**ELO Rating** — sistema de classificação de força relativa entre os times, atualizado partida a partida (K-factor 20, rating inicial 1500).

**Variáveis de diferença** — comparações diretas entre mandante e visitante (diferença de pontos, de forma recente, de gols, de ELO, de desempenho casa/fora) — 9 variáveis adicionais, totalizando **28 variáveis** usadas pelos modelos.

---

## 🔬 Metodologia

O pipeline seguiu uma ordem deliberada, cada etapa validando a anterior antes de avançar:

1. **Limpeza e engenharia de atributos** — com auditoria explícita contra vazamento de dados temporal (toda feature é calculada em ordem cronológica, guardando o valor *antes* de atualizar o histórico com o resultado da partida).
2. **Divisão temporal, nunca aleatória** — treino e teste divididos por data, nunca embaralhados, para simular o uso real do modelo (prever o futuro com base no passado).
3. **Baseline probabilístico** — um modelo "ingênuo" que apenas usa a frequência histórica de cada resultado, como referência mínima que qualquer modelo de Machine Learning precisa superar.
4. **Comparação de modelos** — Random Forest e Regressão Logística treinados e avaliados lado a lado.
5. **Validação probabilística** — Brier Score e curvas de calibração, não só acurácia (ver seção Validação).
6. **Walk-Forward Validation** — `src/models/validation.py` pode comparar Baseline, Random Forest e Regressão Logística por temporada. Ela não calcula Poisson nem Ensemble.
7. **Ajuste orientado por evidência** — o `class_weight="balanced"` da Regressão Logística foi removido depois que o Walk-Forward mostrou que ele piorava a calibração da classe majoritária (ver Validação).
8. **Ensemble de previsão** — `src/models/predict.py` calcula a média aritmética das probabilidades da Regressão Logística e do Poisson. A validação Walk-Forward atual não avalia essa combinação.
9. **Modelo de produção** — a Regressão Logística é retreinada com **100% da base disponível**. Esse ajuste serve para previsão; não substitui uma avaliação fora da amostra em partidas futuras.

---

## 🤖 Modelos

| Modelo | Papel |
|---|---|
| **Baseline** | Frequência histórica de cada resultado — referência mínima |
| **Random Forest** | 200 árvores, testado e descartado para o resultado final (ver Resultados) |
| **Regressão Logística** | Modelo de classificação final, `class_weight=None` (decisão validada no Walk-Forward) |
| **Poisson** | Gols esperados (ataque/defesa por mando de campo) convertidos em probabilidade de resultado via distribuição de Poisson exata |
| **Ensemble** | Média aritmética das probabilidades da Regressão Logística e do Poisson, calculada em `predict.py`; a validação Walk-Forward atual não compara o Ensemble |

---

## ✅ Validação

Acurácia sozinha não mede se as probabilidades são confiáveis. Por isso, a validação central deste projeto usa:

- **Brier Score** — erro quadrático médio entre a probabilidade prevista e o resultado real (quanto menor, melhor; 0 = perfeito).
- **Curva de calibração** — compara probabilidades previstas com frequências observadas. O treino atual gera essa análise para vitória do mandante no split único.
- **Walk-Forward Validation** — `validation.py` testa Baseline, Random Forest e Regressão Logística, usando temporadas anteriores para prever a seguinte. Não inclui Poisson ou Ensemble.

Os valores abaixo são um snapshot de um split temporal único com 9.062 partidas, executado antes da correção que passou a tratar partidas simultâneas em grupo. Eles não são benchmark do código atual. Rode `python main.py` para regenerar métricas após a correção. A validação Walk-Forward não foi executada nessa rodada.

---

## 📈 Resultados

### Split temporal único (80% treino / 20% teste, 1.813 jogos)

| Modelo | Acurácia | Log Loss | Brier Score |
|---|---:|---:|---:|
| Baseline | 46,44% | 1,0647 | 0,6430 |
| Random Forest | 42,69% | 1,0765 | 0,6510 |
| **Regressão Logística** | **47,77%** | **1,0386** | **0,6243** |

Neste split, a Regressão Logística superou o Baseline em Log Loss e Brier Score; o Random Forest ficou abaixo do Baseline nas três métricas mostradas. Acurácia e métricas probabilísticas medem aspectos diferentes do desempenho.

---

## 🔮 Exemplo de Previsão

Exemplo real de saída da interface (os números variam a cada atualização da base, já que o modelo de produção é sempre retreinado com os dados mais recentes):

**Flamengo (mandante) x Palmeiras (visitante)**

| Resultado | Probabilidade |
|---|---:|
| Vitória do Flamengo | 43,19% |
| Empate | 30,48% |
| Vitória do Palmeiras | 26,33% |

**Previsão:** Vitória do Flamengo · **Probabilidade do resultado mais provável:** 43,19% (snapshot anterior à correção de partidas simultâneas)

**Gols esperados:** Flamengo 1,46 · Palmeiras 1,07

> As probabilidades representam a estimativa do modelo com base em dados históricos. Futebol tem variância alta — mesmo um modelo bem calibrado erra a previsão mais provável em boa parte dos jogos (ver Limitações).

---

## ⚠️ Limitações

- **Previsão esportiva é probabilística, não determinística.** Um modelo bem calibrado que diz "60% de chance de vitória" está *certo* mesmo quando o time perde — 40% de chance ainda é uma chance real. Nenhuma saída deste sistema deve ser lida como garantia de resultado.
- **Dados limitados a resultados.** O modelo não inclui transferências, lesões, escalações ou mudanças de técnico. As partidas de 2026 vêm de uma fonte pública separada e só entram no treino depois de terem placar final.
- **Horário das partidas importadas.** A fonte pública registra data, mas não horário; partidas do mesmo dia são tratadas como simultâneas na geração das features.
- **Classes minoritárias são mais difíceis.** O modelo acerta vitórias do mandante com mais frequência do que empates ou vitórias do visitante (F1 Macro em torno de 0,34-0,40) — reflexo direto do desbalanceamento natural do futebol, não um defeito de implementação.
- **O Poisson assume independência entre os gols dos dois times**, uma simplificação comum em modelos de futebol, mas que não captura efeitos como um time "segurar o resultado" no fim de um jogo já decidido.
- **Avaliação ainda limitada.** Os resultados publicados são de um split temporal único; não constituem validação atual do Ensemble nem avaliação em apostas reais.

---

## 🧪 Testes

O projeto tem uma suíte de **63 testes automatizados** (`pytest`), cobrindo:

- Testes unitários de cada módulo (`loader`, `cleaner`, `features`, `metrics`, `poisson_model`, `monte_carlo`, `train_model`, `predict`)
- Casos extremos (item 75 do roadmap de desenvolvimento): arquivo de dados não encontrado, time inexistente, time com poucos jogos registrados
- Testes de integração de ponta a ponta: pipeline de dados, pipeline de treinamento, pipeline de previsão completo

```
pytest tests/ -v
```

---

## 🛠️ Tecnologias

- Python 3.13
- Pandas, NumPy
- Scikit-learn (Random Forest, Regressão Logística)
- python-dotenv (chave da API-Sports)
- SciPy (distribuição de Poisson)
- Streamlit (interface)
- Matplotlib (visualizações)
- Pytest (testes automatizados)

---

## 📁 Estrutura do Projeto

```text
Projeto_Previsao_Futebol/
│
├── dados/
│   ├── raw/                      # CSV original
│   └── processed/                # CSV após limpeza
│
├── modelos/                      # Modelos treinados (.joblib)
├── resultados/                   # Gráficos gerados (Fase 16)
│
├── src/
│   ├── config.py                 # Parâmetros e caminhos centralizados
│   │
│   ├── data/
│   │   ├── loader.py             # CSV histórico + jogos encerrados no SQLite
│   │   ├── cleaner.py            # Limpeza e variável alvo
│   │   ├── importar_jogos_2026.py # Importação da fonte pública de 2026
│   │   └── atualizar_partidas.py # Importação pela API-Sports (sujeita ao plano)
│   │
│   ├── features/
│   │   └── form.py               # Forma recente, gols, casa/fora, ELO
│   │
│   ├── models/
│   │   ├── metrics.py            # Brier Score, calibração
│   │   ├── poisson_model.py      # Probabilidades via Poisson exato
│   │   ├── monte_carlo.py        # Simulação de resultado e placares
│   │   ├── train_model.py        # Treino, validação e modelo de produção
│   │   ├── predict.py            # Previsão (Regressão Logística + Poisson)
│   │   └── validation.py         # Walk-Forward Validation
│   │
│   └── visualizations.py         # Gráficos (Fase 16)
│
├── tests/                        # Suíte de testes (pytest)
│
├── app.py                        # Interface Streamlit
├── main.py                       # Pipeline de console
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar o pipeline completo no console
# (validação + modelo de produção + previsão de exemplo)
python main.py

# Rodar a interface web
streamlit run app.py

# Rodar a suíte de testes
pytest tests/ -v

# Gerar os gráficos (Fase 16)
python -m src.visualizations

# Rodar a validação Walk-Forward (o número de temporadas depende da base)
python -m src.models.validation
```