"""
================================================================================
Detecção de Cédulas Falsas com Rede Neural (implementação do zero em numpy)
================================================================================

Projeto originalmente desenvolvido para a disciplina de Computação Evolucionária
e Conexionista (UFABC), implementando uma rede neural feedforward simples para
classificar cédulas bancárias como autênticas ou falsas.

DATASET:
    Banknote Authentication Dataset - características extraídas de imagens
    de cédulas através da Transformada de Wavelet.

REFERÊNCIA:
    V. Lohweg. "Banknote Authentication," UCI Machine Learning Repository, 2012.
    Disponível em: https://doi.org/10.24432/C55P57
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem display, necessário para rodar em servidor/CI
import matplotlib.pyplot as plt

from data_loader import load_dataset, train_test_split_manual, zscore_normalize
from metrics import calcular_metricas
from neural_network import RedeNeural

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "data_banknote_authentication.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina uma rede neural (numpy puro) para detectar cédulas falsas."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quiet", action="store_true", help="Não imprime o progresso a cada época."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dados -----------------------------------------------------------------
    df = load_dataset(args.data_path)
    print("Informações do dataset:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, test_size=args.test_size, seed=args.seed
    )
    X_train_scaled, X_test_scaled = zscore_normalize(X_train, X_test)
    print(f"\nTreino: {X_train_scaled.shape}, Teste: {X_test_scaled.shape}")

    # 2. Treino ------------------------------------------------------------------
    # Não reseedamos aqui de propósito: os pesos iniciais seguem o estado
    # aleatório deixado pelo split acima, preservando a reprodutibilidade
    # de ponta a ponta a partir de um único --seed.
    modelo = RedeNeural(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        output_size=1,
        lr=args.lr,
    )
    modelo.train(X_train_scaled, y_train, epochs=args.epochs, verbose=not args.quiet)

    # 3. Predição e métricas -------------------------------------------------------
    probabilidades = modelo.forward(X_test_scaled)
    y_pred = (probabilidades > 0.5).astype(int)
    metricas = calcular_metricas(y_test, y_pred)

    resumo = (
        f"Acurácia: {metricas['acuracia']:.4f} | "
        f"Precisão: {metricas['precisao']:.4f} | "
        f"Recall: {metricas['recall']:.4f} | "
        f"F1-Score: {metricas['f1_score']:.4f}"
    )
    print(resumo)

    resultado_path = args.output_dir / "resultado.txt"
    resultado_path.write_text(resumo + "\n", encoding="utf-8")

    # 4. Gráfico de convergência ---------------------------------------------------
    plt.plot(modelo.loss_history)
    plt.title("Convergência do Erro (Loss)")
    plt.xlabel("Época")
    plt.ylabel("Erro")
    plt.grid(True)
    plt.savefig(args.output_dir / "loss_curve.png")

    print(f"\nResultados salvos em: {resultado_path}")
    print(f"Gráfico salvo em: {args.output_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
