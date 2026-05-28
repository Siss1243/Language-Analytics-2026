"""
train_model.py

Train and save a KeyNMF topic model using preprocessed StorySeeker data.
"""

# IMPORTS
import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from turftopic import KeyNMF


# PATH HANDLING
ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "in"
OUT_DIR = ROOT / "out"

PROCESSED_DATA_DIR = IN_DIR / "processed"
MODEL_DIR = OUT_DIR / "models"


# FUNCTIONS: MAIN

def main(args):
    print("Running train_model.py...")

    ensure_dir(MODEL_DIR)

    data_path = resolve_data_path(args.data_path)

    df = load_preprocessed_data(data_path)

    validate_training_data(df)

    model = train_keynmf_model(
        texts=df["processed_text"].tolist(),
        n_topics=args.n_topics,
        random_state=args.random_state,
    )

    model_path = build_model_path(
        n_topics=args.n_topics,
        remove_stopwords=args.remove_stopwords,
        sample_size=args.sample_size,
        run_name=args.run_name,
    )

    save_model(model, model_path)

    save_training_metadata(
        model_path=model_path,
        data_path=data_path,
        n_topics=args.n_topics,
        remove_stopwords=args.remove_stopwords,
        sample_size=args.sample_size,
        random_state=args.random_state,
        run_name=args.run_name,
    )


# FUNCTIONS: SUPPORT

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_data_path(data_path):
    """
    Use user-specified data path if provided.
    Otherwise, use the newest processed CSV file.
    """

    if data_path is not None:
        return Path(data_path)

    candidates = sorted(PROCESSED_DATA_DIR.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(
            "No processed data files found. Run build_data.py first."
        )

    return candidates[-1]


def load_preprocessed_data(path):
    df = pd.read_csv(path)
    print(f"Loaded preprocessed data from: {path}")
    print("Shape:", df.shape)

    return df


def validate_training_data(df):
    required_cols = {"processed_text", "label"}

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["processed_text"].isna().any():
        raise ValueError("processed_text contains missing values.")


def train_keynmf_model(texts, n_topics=10, random_state=42):
    print("\nTRAINING KEYNMF MODEL")
    print("---------------------")
    print(f"Documents: {len(texts)}")
    print(f"Topics: {n_topics}")

    model = KeyNMF(
        n_components=n_topics,
        random_state=random_state,
    )

    model.fit(texts)

    print("\nModel training complete.")

    return model


def build_model_path(
    n_topics,
    remove_stopwords=False,
    sample_size=None,
    run_name=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    stop = int(remove_stopwords)
    sample = sample_size if sample_size is not None else "full"

    base_name = (
        f"keynmf_k{n_topics}"
        f"_stop{stop}"
        f"_sample{sample}"
        f"_{timestamp}"
    )

    if run_name is not None:
        base_name = f"{run_name}_{base_name}"

    return MODEL_DIR / f"{base_name}.joblib"


def save_model(model, path):
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")


def save_training_metadata(
    model_path,
    data_path,
    n_topics,
    remove_stopwords,
    sample_size,
    random_state,
    run_name,
):
    metadata = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "n_topics": n_topics,
        "remove_stopwords": remove_stopwords,
        "sample_size": sample_size,
        "random_state": random_state,
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    metadata_path = model_path.with_suffix(".metadata.csv")

    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)

    print(f"Metadata saved to: {metadata_path}")


# FUNCTIONS: PARSER

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a KeyNMF model on preprocessed StorySeeker data."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to preprocessed CSV. "
            "If omitted, newest file in in/processed/ is used."
        ),
    )

    parser.add_argument(
        "--n-topics",
        type=int,
        default=10,
        help="Number of topics to train.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size used in the processed data, for filename tracking.",
    )

    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Whether the processed data used stopword removal.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for model fitting.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom prefix for saved model files.",
    )

    return parser.parse_args()


# CALL FOR MAIN
if __name__ == "__main__":
    args = parse_args()
    main(args)
