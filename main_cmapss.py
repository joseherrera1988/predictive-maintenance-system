"""
Entry point for CMAPSS FD001 RUL prediction experiments.

Usage:
    python main_cmapss.py --model ml
    python main_cmapss.py --model dl
    python main_cmapss.py --model both
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="CMAPSS FD001 RUL Prediction")
    parser.add_argument(
        "--model",
        choices=["ml", "dl", "both"],
        default="both",
        help="Model to train: ml=Random Forest, dl=LSTM, both=run both (default: both)",
    )
    parser.add_argument("--train", default="data/raw/train_FD001.txt", help="Path to train_FD001.txt")
    parser.add_argument("--test", default="data/raw/test_FD001.txt", help="Path to test_FD001.txt")
    parser.add_argument("--rul", default="data/raw/RUL_FD001.txt", help="Path to RUL_FD001.txt")

    # ML hyperparams
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)

    # DL hyperparams
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    if args.model in ("ml", "both"):
        print("\n" + "=" * 50)
        print("  Random Forest Regressor")
        print("=" * 50)
        from src.train_ml_rul import train as train_ml
        train_ml(
            train_path=args.train,
            test_path=args.test,
            rul_path=args.rul,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    if args.model in ("dl", "both"):
        print("\n" + "=" * 50)
        print("  LSTM Regressor")
        print("=" * 50)
        from src.train_dl_rul import train as train_dl
        train_dl(
            train_path=args.train,
            test_path=args.test,
            rul_path=args.rul,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    print("\n[main_cmapss] Done.")


if __name__ == "__main__":
    main()
