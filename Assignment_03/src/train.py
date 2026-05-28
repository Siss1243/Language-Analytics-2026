'''
train.py

Loads Taylor Swift lyrics from CSV, preprocesses the text data,
trains an n-gram language model, and saves the trained model.
'''


#IMPORTS
from pathlib import Path
import pandas as pd
import argparse

from ngrammodel import NgramModel

#PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "in" / "TaylorSwift.csv"
OUTPUT_DIR = PROJECT_ROOT / "out" / "models"

#FUNCTIONS: MAIN
def train(args):
    """Main training pipeline."""
    
    df = load_data(args.input)

    lyrics = extract_lyrics(df,
        min_words=args.min_words,
        min_chars=args.min_chars,
        max_chars=args.max_chars)

    print(f"\nLoaded {len(lyrics)} lyrics for training.")

    if not lyrics:
        raise ValueError("No lyrics remain after filtering.")

    model = NgramModel(
        name=args.model_name,
        ngram_size=args.ngram_size)
    
    model.train(lyrics)

    if args.output is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save()
    else:
        model.save(str(args.output))


#FUNCTIONS: SUPPORT
def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("\n--- DATASET LOADED ---")
    print(f"File: {csv_path}")
    print(f"Rows in raw dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df

def extract_lyrics(df, min_words, min_chars, max_chars):
    """
    Filter the dataframe and return lyric texts as a list of strings.
    """
    working_df = df.dropna(subset=["Lyric"]).copy()
    working_df["Lyric"] = working_df["Lyric"].astype(str)

    # Remove unreleased songs
    unreleased_mask = (
        working_df["Album"]
        .fillna("")
        .str.strip()
        .str.lower()
        == "unreleased songs")

    working_df = working_df.loc[~unreleased_mask].copy()

    # Length features
    working_df["word_count"] = working_df["Lyric"].str.split().str.len()
    working_df["char_count"] = working_df["Lyric"].str.len()

    # Apply filters
    working_df = working_df.loc[working_df["word_count"] >= min_words].copy()
    working_df = working_df.loc[working_df["char_count"] >= min_chars].copy()
    working_df = working_df.loc[working_df["char_count"] <= max_chars].copy()

    return working_df["Lyric"].tolist()



#FUNCTIONS: PARSER
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train an n-gram model on filtered Taylor Swift lyrics."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional full output path for the trained model (.pkl). "
             "If omitted, the model uses its default timestamped save path.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="taylor",
        help="Name of the trained model.",
    )

    parser.add_argument(
        "--ngram-size",
        type=int,
        default=3,
        help="Size of n-grams to train (e.g. 2 for bigram, 3 for trigram).",
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        help="Minimum number of words required for a lyric to be kept.",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Minimum number of characters required for a lyric to be kept.",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=2825,
        help="Maximum number of characters allowed for a lyric to be kept.",
    )

    return parser.parse_args()

# CALL MAIN
if __name__ == "__main__":
    args = parse_args()
    train(args)