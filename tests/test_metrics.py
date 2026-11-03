import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import calcular_metricas
from neural_network import sigmoid, sigmoid_derivative
from data_loader import train_test_split_manual


def test_calcular_metricas_caso_perfeito():
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0], [1], [1], [0]])

    m = calcular_metricas(y_true, y_pred)

    assert m["acuracia"] == 1.0
    assert m["precisao"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0


def test_calcular_metricas_caso_misto():
    y_true = np.array([[1], [1], [0], [0]])
    y_pred = np.array([[1], [0], [0], [1]])  # 1 TP, 1 FN, 1 TN, 1 FP

    m = calcular_metricas(y_true, y_pred)

    assert m["acuracia"] == 0.5
    assert m["precisao"] == 0.5
    assert m["recall"] == 0.5


def test_sigmoid_intervalo():
    z = np.array([-100.0, 0.0, 100.0])
    s = sigmoid(z)

    assert np.all((s >= 0) & (s <= 1))
    assert np.isclose(s[1], 0.5)


def test_sigmoid_derivative_maximo_em_zero():
    # A derivada da sigmoid é máxima (0.25) em z = 0
    assert np.isclose(sigmoid_derivative(np.array([0.0]))[0], 0.25)


def test_train_test_split_manual_proporcoes():
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert y_train.shape[0] == 80
    assert y_test.shape[0] == 20


def test_train_test_split_manual_reprodutivel():
    X = np.arange(50).reshape(-1, 1)
    y = np.arange(50).reshape(-1, 1)

    r1 = train_test_split_manual(X, y, seed=42)
    r2 = train_test_split_manual(X, y, seed=42)

    for a, b in zip(r1, r2):
        assert np.array_equal(a, b)
