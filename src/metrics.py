"""Métricas de avaliação para classificação binária, a partir da matriz de confusão."""

from __future__ import annotations

import numpy as np


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calcula acurácia, precisão, recall e F1-score a partir da matriz de confusão.

        Acurácia:  (TP + TN) / Total
        Precisão:  TP / (TP + FP)
        Recall:    TP / (TP + FN)
        F1-Score:  média harmônica entre precisão e recall
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acuracia = (tp + tn) / len(y_true)
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precisao * recall) / (precisao + recall)
        if (precisao + recall) > 0
        else 0.0
    )

    return {
        "acuracia": float(acuracia),
        "precisao": float(precisao),
        "recall": float(recall),
        "f1_score": float(f1),
    }
