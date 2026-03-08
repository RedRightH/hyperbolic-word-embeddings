import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "trained_models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

EUCLIDEAN_CONFIG = {
    'embedding_dim': 100,
    'window_size': 5,
    'min_count': 1,
    'workers': 4,
    'epochs': 50,
    'learning_rate': 0.025,
    'seed': 42
}

HYPERBOLIC_CONFIG = {
    'embedding_dim': 10,
    'learning_rate': 0.1,
    'epochs': 300,
    'batch_size': 32,
    'negative_samples': 10,
    'seed': 42,
    'burn_in': 10,
    'burn_in_lr': 0.01
}

EVALUATION_CONFIG = {
    'test_split': 0.2,
    'k_values': [1, 5, 10],
    'seed': 42
}
