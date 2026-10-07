# ⚽ Brasileirão Predictor

Sistema de previsão probabilística de partidas do Campeonato Brasileiro de futebol, construído em Python com uma metodologia de validação estatística rigorosa — não apenas um modelo treinado, mas um pipeline onde cada decisão (escolha de modelo, hiperparâmetros, combinação de modelos) foi testada contra dados reais antes de ser adotada.

O sistema estima as probabilidades de:

- Vitória do mandante
- Empate
- Vitória do visitante

E complementa com gols esperados e os placares mais prováveis, via simulação de Monte Carlo.

---

## 🎯 Objetivo

Construir um sistema de previsão de partidas do Campeonato Brasileiro que não apenas gera probabilidades, mas **comprova que essas probabilidades são confiáveis** — avaliadas por calibração (Brier Score) e estabilidade ao longo do tempo (Walk-Forward Validation em 17 temporadas), não só por acurácia num único recorte de teste.

---

## 📊 Dados

- **8.785 partidas** do Campeonato Brasileiro, 2003 a 2024 (21 temporadas)
- Colunas originais: data, mandante, visitante, placar, arena, estado dos times
- Distribuição de resultados:

| Resultado | Partidas | Percentual |
|---|---:|---:|
| Vitória do mandante | 4.359 | 49,62% |
| Empate | 2.321 | 26,42% |
| Vitória do visitante | 2.105 | 23,96% |

A vantagem de jogar em casa é a informação mais forte da base — quase metade dos jogos termina em vitória do mandante.

---

## 🧠 Engenharia de Atributos

Todas as variáveis abaixo são calculadas **usando apenas informação anterior à partida** — nenhuma usa resultado da própria partida ou de jogos futuros (auditado explicitamente, ver seção Metodologia).

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
6. **Walk-Forward Validation** — a mesma comparação repetida em 17 temporadas diferentes (2008-2024), para confirmar que o resultado não foi sorte de um único recorte de teste.
7. **Ajuste orientado por evidência** — o `class_weight="balanced"` da Regressão Logística foi removido depois que o Walk-Forward mostrou que ele piorava a calibração da classe majoritária (ver Validação).
8. **Modelo híbrido (Ensemble)** — Regressão Logística combinada com um modelo de Poisson independente, só mantida porque o Walk-Forward confirmou que o Ensemble reduz o erro médio em relação aos dois modelos isolados.
9. **Modelo de produção** — depois de validada a metodologia, o modelo final é retreinado com **100% dos dados** (não só os 80% de treino), já que não existe mais teste a fazer, apenas previsão real.

---

## 🤖 Modelos

| Modelo | Papel |
|---|---|
| **Baseline** | Frequência histórica de cada resultado — referência mínima |
| **Random Forest** | 200 árvores, testado e descartado para o resultado final (ver Resultados) |
| **Regressão Logística** | Modelo de classificação final, `class_weight=None` (decisão validada no Walk-Forward) |
| **Poisson** | Gols esperados (ataque/defesa por mando de campo) convertidos em probabilidade de resultado via distribuição de Poisson exata |
| **Ensemble** | Média das probabilidades da Regressão Logística e do Poisson — **o modelo que efetivamente gera a previsão final** |

---

## ✅ Validação

Acurácia sozinha não mede se as probabilidades são confiáveis. Por isso, a validação central deste projeto usa:

- **Brier Score** — erro quadrático médio entre a probabilidade prevista e o resultado real (quanto menor, melhor; 0 = perfeito).
- **Curva de calibração** — quando o modelo diz "60% de chance", isso precisa acontecer perto de 60% das vezes de verdade.
- **Walk-Forward Validation** — treina com temporadas passadas, testa na seguinte, avança, repete — 17 temporadas testadas (2008 a 2024), em vez de um único corte 80/20.

**Achado principal da validação:** a Regressão Logística com `class_weight="balanced"` (configuração inicial) subestimava sistematicamente a probabilidade da classe majoritária (Mandante) — confirmado pela curva de calibração. Removendo o balanceamento, o Brier Score médio no Walk-Forward caiu de 0,658 para 0,615, e a calibração ficou visivelmente mais próxima da linha ideal.

**Achado sobre o Ensemble:** no Walk-Forward, o Ensemble (Regressão Logística + Poisson) teve o menor Brier Score médio entre as três abordagens e venceu a Regressão Logística isolada em 10 das 17 temporadas testadas — por isso é o modelo de produção, não uma combinação adotada "porque parece mais sofisticado".

---

## 📈 Resultados

### Split único (80% treino / 20% teste, 1.757 jogos)

| Modelo | Acurácia | Log Loss | Brier Score |
|---|---:|---:|---:|
| Baseline | 45,99% | 1,0688 | 0,6459 |
| Random Forest | ~43-47%* | ~1,05-1,07* | ~0,632-0,646* |
| **Regressão Logística** | **47,81%** | **1,0404** | **0,6254** |

\* *O Random Forest mostrou variação perceptível entre versões do scikit-learn mesmo com `random_state` fixo — um lembrete de que reprodutibilidade exata entre ambientes nunca é garantida em bibliotecas de ML. Em nenhum ambiente testado ele superou a Regressão Logística com folga.*

### Walk-Forward Validation (17 temporadas, 2008-2024)

| Abordagem | Brier médio | Desvio-padrão | Temporadas em que venceu |
|---|---:|---:|---:|
| **Ensemble (Regressão Logística + Poisson)** | **0,6139** | 0,0185 | **10 de 17** |
| Regressão Logística | 0,6152 | 0,0209 | 6 de 17 |
| Poisson | 0,6214 | 0,0177 | 1 de 17 |

O Ensemble é o modelo de produção: melhor média, segunda menor variabilidade entre temporadas, e vitória na maioria dos anos testados.

---

## 🔮 Exemplo de Previsão

Exemplo real de saída da interface (os números variam a cada atualização da base, já que o modelo de produção é sempre retreinado com os dados mais recentes):

**Flamengo (mandante) x Palmeiras (visitante)**

| Resultado | Probabilidade |
|---|---:|
| Vitória do Flamengo | 45,2% |
| Empate | 28,1% |
| Vitória do Palmeiras | 26,7% |

**Previsão:** Vitória do Flamengo · **Confiança:** 45,2%

**Gols esperados:** Flamengo 1,46 · Palmeiras 1,07

> As probabilidades representam a estimativa do modelo com base em dados históricos. Futebol tem variância alta — mesmo um modelo bem calibrado erra a previsão mais provável em boa parte dos jogos (ver Limitações).

---

## ⚠️ Limitações

- **Previsão esportiva é probabilística, não determinística.** Um modelo bem calibrado que diz "60% de chance de vitória" está *certo* mesmo quando o time perde — 40% de chance ainda é uma chance real. Nenhuma saída deste sistema deve ser lida como garantia de resultado.
- **Dados anteriores a cada atualização.** O modelo de produção é retreinado a cada execução com os dados disponíveis no CSV local — ele não reflete transferências, lesões, mudanças de técnico ou qualquer evento fora do histórico de resultados em si.
- **Classes minoritárias são mais difíceis.** O modelo acerta vitórias do mandante com mais frequência do que empates ou vitórias do visitante (F1 Macro em torno de 0,34-0,40) — reflexo direto do desbalanceamento natural do futebol, não um defeito de implementação.
- **O Poisson assume independência entre os gols dos dois times**, uma simplificação comum em modelos de futebol, mas que não captura efeitos como um time "segurar o resultado" no fim de um jogo já decidido.
- **Validado estatisticamente, não testado em apostas reais.** Brier Score e calibração medem qualidade estatística das probabilidades, não retorno financeiro — são coisas relacionadas, mas não idênticas.

---

## 🧪 Testes

O projeto tem uma suíte de **62 testes automatizados** (`pytest`), cobrindo:

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
│   │   ├── loader.py             # Carregamento do CSV
│   │   └── cleaner.py            # Limpeza e variável alvo
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

# Rodar a validação Walk-Forward (17 temporadas — demora alguns minutos)
python -m src.models.validation
```