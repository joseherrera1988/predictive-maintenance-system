import argparse
from src.data_pipeline import run_pipeline
from src.train_ml import train as train_ml
from src.train_dl import train as train_dl


def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance AI")
    parser.add_argument("--data", type=str, required=True, help="Raw CSV filename in data/raw/")
    parser.add_argument(
        "--model",
        choices=["ml", "dl", "both"],
        default="ml",
        help="Which model to train (default: ml)",
    )
    args = parser.parse_args()

    print(f"[main] Loading and preprocessing {args.data} ...")
    df = run_pipeline(args.data)

    if args.model in ("ml", "both"):
        print("[main] Training ML model ...")
        train_ml(df)

    if args.model in ("dl", "both"):
        print("[main] Training DL model ...")
        train_dl(df)

    print("[main] Done.")


if __name__ == "__main__":
    main()
