"""
Rede neural feedforward com uma camada oculta, implementada do zero (numpy),
treinada com backpropagation e gradient descent.

Arquitetura:
    Entrada (4 features) -> Oculta (sigmoid) -> Saída (sigmoid, 1 neurônio)
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid: σ(z) = 1 / (1 + e^-z). Mapeia valores para (0, 1)."""
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivada da sigmoid: σ'(z) = σ(z) * (1 - σ(z)). Usada no backprop."""
    s = sigmoid(z)
    return s * (1 - s)


class RedeNeural:
    """
    Rede neural feedforward com uma camada oculta.
    Treinada com backpropagation e otimizada via gradient descent.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        lr: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.lr = lr

        if seed is not None:
            np.random.seed(seed)
        # Pesos: valores pequenos e aleatórios (quebra a simetria entre neurônios)
        # Bias: inicializado em zero
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.loss_history: list[float] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Propaga os dados pela rede: Entrada -> Oculta -> Saída."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Calcula gradientes via regra da cadeia e atualiza os pesos."""
        m = y.shape[0]

        # Gradientes da camada de saída
        dz2 = output - y
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Gradientes da camada oculta (propagação do erro através de W2)
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(self.z1)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent: w_novo = w_atual - lr * gradiente
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, verbose: bool = True) -> None:
        """Loop de treinamento: forward -> backward por N épocas."""
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            # Binary Cross-Entropy: -[y*log(ŷ) + (1-y)*log(1-ŷ)]
            loss = -np.mean(
                y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8)
            )
            self.loss_history.append(float(loss))

            if verbose and i % 100 == 0:
                print(f"Época {i}, Erro: {loss:.4f}")
