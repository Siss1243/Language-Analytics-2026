"""
build_data.py

Prepare and preprocess the StorySeeker dataset for topic modeling.
"""

# IMPORTS
import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# PATH HANDLING
ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "in"
OUT_DIR = ROOT / "out"

RAW_DATA_PATH = IN_DIR / "storyseeker.csv"
PROCESSED_DATA_DIR = IN_DIR / "processed"


# FUNCTIONS: MAIN

def main(args):
    print("Running build_data.py...")

    ensure_dir(PROCESSED_DATA_DIR)

    df = load_data(RAW_DATA_PATH)

    df = prepare_dataset(
        df=df,
        text_col=args.text_col,
        label_col=args.label_col,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    inspect_prepared_data(df)

    df = preprocess_dataset(
        df=df,
        remove_stopwords=args.remove_stopwords,
    )

    inspect_preprocessed_data(df)

    output_path = build_processed_data_path(
        remove_stopwords=args.remove_stopwords,
        sample_size=args.sample_size,
    )

    save_data(df, output_path)


# FUNCTIONS: SUPPORT

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_data(path):
    return pd.read_csv(path)


def prepare_dataset(
    df,
    text_col="text",
    label_col="gold_consensus",
    sample_size=None,
    random_state=42,
):
    df = df[[text_col, label_col]].copy()

    df = df.rename(
        columns={
            text_col: "text",
            label_col: "label",
        }
    )

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)

    label_map = {
        0: "no_story",
        1: "story",
    }

    df["label"] = df["label"].map(label_map)

    if df["label"].isna().any():
        raise ValueError(
            "Some labels could not be mapped. "
            "Check whether the label column only contains 0 and 1."
        )

    df["char_length"] = df["text"].str.len()
    df["word_length"] = df["text"].str.split().str.len()

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    df = df.reset_index(drop=True)

    return df


def inspect_prepared_data(df):
    print("\nPREPARED DATASET OVERVIEW")
    print("-------------------------")
    print("Shape:", df.shape)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nLabel distribution (%):")
    print(round(df["label"].value_counts(normalize=True) * 100, 2))

    print("\nOverall text length:")
    print(
        df[["char_length", "word_length"]]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )

    print("\nText length by label:")
    print(
        df.groupby("label")[["char_length", "word_length"]]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )


def preprocess_dataset(df, remove_stopwords=False):
    df = df.copy()

    df["processed_text"] = df["text"].apply(clean_text)

    if remove_stopwords:
        df["processed_text"] = df["processed_text"].apply(
            remove_stopwords_from_text
        )

    df["processed_char_length"] = df["processed_text"].str.len()
    df["processed_word_length"] = df["processed_text"].str.split().str.len()

    return df


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove Reddit TL;DR markers
    text = re.sub(r"\btl\s*;?\s*dr\b", " ", text)
    text = re.sub(r"\btl:dr\b", " ", text)

    # Remove Reddit/Markdown emphasis markers
    text = re.sub(r"\*", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_stopwords_from_text(text):
    tokens = text.split()

    tokens = [
        token for token in tokens
        if token.lower() not in ENGLISH_STOP_WORDS
    ]

    return " ".join(tokens)


def inspect_preprocessed_data(df):
    print("\nPREPROCESSED DATASET OVERVIEW")
    print("-----------------------------")

    print("\nOverall processed text length:")
    print(
        df[["processed_char_length", "processed_word_length"]]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )

    print("\nProcessed text length by label:")
    print(
        df.groupby("label")[["processed_char_length", "processed_word_length"]]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )

    print(
        df.groupby("label")["processed_word_length"]
        .agg(["mean", "std"])
        .round(2))

    print("\nLexical diversity overall:")
    print(pd.Series(calculate_lexical_stats(df["processed_text"])))

    print("\nLexical diversity by label:")
    rows = []

    for label, subset in df.groupby("label"):
        stats = calculate_lexical_stats(subset["processed_text"])
        stats["label"] = label
        rows.append(stats)

    lexical_df = pd.DataFrame(rows)
    print(
        lexical_df[
            ["label", "total_tokens", "unique_tokens", "type_token_ratio"]
        ]
    )

    print("\nExample processed text:")
    print(df["processed_text"].iloc[0][:500])


def tokenize_for_descriptives(text):
    text = text.lower()
    return re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)


def calculate_lexical_stats(texts):
    all_tokens = []

    for text in texts:
        all_tokens.extend(tokenize_for_descriptives(text))

    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": round(ttr, 4),
    }


def build_processed_data_path(remove_stopwords=False, sample_size=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stop = int(remove_stopwords)
    sample = sample_size if sample_size is not None else "full"

    filename = f"storyseeker_processed_stop{stop}_sample{sample}_{timestamp}.csv"

    return PROCESSED_DATA_DIR / filename


def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"\nProcessed data saved to: {path}")


# FUNCTIONS: PARSER

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare and preprocess StorySeeker data."
    )

    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Name of the column containing document text.",
    )

    parser.add_argument(
        "--label-col",
        type=str,
        default="gold_consensus",
        help="Name of the column containing story/no-story labels.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for faster experimentation.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )

    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove English stopwords during preprocessing.",
    )

    return parser.parse_args()


# CALL FOR MAIN
if __name__ == "__main__":
    args = parse_args()
    main(args)