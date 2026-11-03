"""
Carregamento, divisão treino/teste e normalização do dataset
Banknote Authentication.

Referência do dataset:
    V. Lohweg. "Banknote Authentication," UCI Machine Learning Repository, 2012.
    https://doi.org/10.24432/C55P57
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMN_NAMES = ["variance", "skewness", "curtosis", "entropy", "class"]


def load_dataset(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um CSV sem cabeçalho."""
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def train_test_split_manual(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
):
    """
    Divide o dataset em treino/teste sem depender do scikit-learn.

    Implementação manual: embaralha os índices com uma seed fixa
    (reprodutibilidade) e separa a fração `test_size` para teste.
    """
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_set_size = int(X.shape[0] * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def zscore_normalize(X_train: np.ndarray, X_test: np.ndarray):
    """
    Normalização Z-Score: z = (x - μ) / σ

    IMPORTANTE: média e desvio padrão são calculados apenas no conjunto
    de treino para evitar data leakage. O termo 1e-8 evita divisão por zero.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled
