"""
Configurações centrais do Brasileirão Predictor.

Centraliza caminhos de arquivos e os principais parâmetros dos
modelos (item 60 do roadmap), para que mudar um valor -- por
exemplo, quantas simulações de Monte Carlo rodar, ou quantos
jogos considerar na forma recente -- não exija caçar o número
espalhado em vários arquivos diferentes.
"""

from pathlib import Path

# ==========================================
# CAMINHOS
# ==========================================

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_DADOS_RAW = RAIZ_PROJETO / "dados" / "raw" / "campeonato-brasileiro-full.csv"
CAMINHO_DADOS_PROCESSED = RAIZ_PROJETO / "dados" / "processed" / "brasileirao_limpo.csv"
CAMINHO_MODELOS = RAIZ_PROJETO / "modelos"
CAMINHO_RESULTADOS = RAIZ_PROJETO / "resultados"

# ==========================================
# FEATURES / JANELA DE FORMA RECENTE
# ==========================================

ULTIMOS_JOGOS_FORMA = 5

# ==========================================
# ELO RATING
# ==========================================

ELO_INICIAL = 1500.0
ELO_K_FACTOR = 20

# ==========================================
# DIVISÃO TREINO/TESTE (split único, train_model.py)
# ==========================================

PROPORCAO_TREINO = 0.80

# ==========================================
# RANDOM FOREST
# ==========================================

RF_N_ESTIMATORS = 200
RF_CLASS_WEIGHT = "balanced"
RF_RANDOM_STATE = 42

# ==========================================
# REGRESSÃO LOGÍSTICA
# ----------------------------------------
# class_weight=None: o Walk-Forward Validation (Fase 11,
# src/models/validation.py) mostrou que "balanced" piorava a
# calibração da classe majoritária (Mandante) e o Brier Score.
# ==========================================

LOGISTICA_MAX_ITER = 3000
LOGISTICA_CLASS_WEIGHT = None
LOGISTICA_RANDOM_STATE = 42

# ==========================================
# MONTE CARLO / POISSON
# ==========================================

N_SIMULACOES_MONTE_CARLO = 10000
MAX_GOLS_POISSON = 10