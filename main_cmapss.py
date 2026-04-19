"""
Entry point for CMAPSS RUL prediction experiments.

Usage:
    python main_cmapss.py --model ml
    python main_cmapss.py --model dl
    python main_cmapss.py --model xgb
    python main_cmapss.py --model all
    python main_cmapss.py --model ml --train data/raw/train_FD002.txt --test data/raw/test_FD002.txt --rul data/raw/RUL_FD002.txt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="CMAPSS RUL Prediction")
    parser.add_argument(
        "--model",
        choices=["ml", "dl", "xgb", "all"],
        default="all",
        help="Model to train: ml=Random Forest, dl=LSTM, xgb=XGBoost, all=run all three (default: all)",
    )
    parser.add_argument("--train", default="data/raw/train_FD001.txt", help="Path to train txt file")
    parser.add_argument("--test", default="data/raw/test_FD001.txt", help="Path to test txt file")
    parser.add_argument("--rul", default="data/raw/RUL_FD001.txt", help="Path to RUL txt file")

    # RF hyperparams
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)

    # XGB hyperparams
    parser.add_argument("--xgb-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)

    # DL hyperparams
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    if args.model in ("ml", "all"):
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

    if args.model in ("xgb", "all"):
        print("\n" + "=" * 50)
        print("  XGBoost Regressor")
        print("=" * 50)
        from src.train_xgb_rul import train as train_xgb
        train_xgb(
            train_path=args.train,
            test_path=args.test,
            rul_path=args.rul,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_lr,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
        )

    if args.model in ("dl", "all"):
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
