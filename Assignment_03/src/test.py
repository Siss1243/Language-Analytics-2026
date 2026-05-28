"""
train.py

Loads Taylor Swift lyrics from CSV, inspects the dataset,
extracts lyric texts, and prepares data for later n-gram training.
"""

# IMPORTS
import argparse
from pathlib import Path

import pandas as pd


# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "in" / "TaylorSwift.csv"
OUTPUT_PATH = PROJECT_ROOT / "out" / "taylor_model.pkl"


# FUNCTIONS: SUPPORT
def vprint(verbose: bool, *args, **kwargs) -> None:
    """Print only when verbose mode is enabled."""
    if verbose:
        print(*args, **kwargs)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def inspect_data(df: pd.DataFrame, verbose: bool = True) -> None:
    """Print a broad overview of the raw dataset for inspection."""
    vprint(verbose, "\n--- DATA OVERVIEW ---")
    vprint(verbose, f"Shape: {df.shape}")
    vprint(verbose, f"Columns: {list(df.columns)}")

    vprint(verbose, "\n--- HEAD ---")
    vprint(verbose, df.head())

    vprint(verbose, "\n--- DATA TYPES ---")
    vprint(verbose, df.dtypes)

    vprint(verbose, "\n--- MISSING VALUES PER COLUMN ---")
    vprint(verbose, df.isna().sum())

    if "Artist" in df.columns:
        vprint(verbose, "\n--- UNIQUE ARTISTS ---")
        vprint(verbose, df["Artist"].value_counts().head(10))

    if "Album" in df.columns:
        vprint(verbose, "\n--- TOP ALBUMS ---")
        vprint(verbose, df["Album"].value_counts().head(10))

    if "Year" in df.columns:
        vprint(verbose, "\n--- YEAR DISTRIBUTION ---")
        vprint(verbose, df["Year"].value_counts().sort_index())

    if "Lyric" in df.columns:
        lyric_series = df["Lyric"].dropna().astype(str)

        lyric_lengths_chars = lyric_series.str.len()
        vprint(verbose, "\n--- LYRIC LENGTHS (CHARACTERS) ---")
        vprint(verbose, lyric_lengths_chars.describe())

        word_counts = lyric_series.str.split().str.len()
        vprint(verbose, "\n--- LYRIC LENGTHS (WORDS) ---")
        vprint(verbose, word_counts.describe())

        vprint(verbose, "\n--- SAMPLE LYRIC SNIPPETS (RAW) ---")
        sample_lyrics = lyric_series.head(3).tolist()
        for i, lyric in enumerate(sample_lyrics, start=1):
            vprint(verbose, f"\nLyric sample {i} (start):")
            vprint(verbose, lyric[:500])

        vprint(verbose, "\n--- POSSIBLE PREPROCESSING SIGNALS ---")
        contains_brackets = lyric_series.str.contains(r"\[.*?\]", regex=True).sum()
        contains_newlines = lyric_series.str.contains(r"\n", regex=True).sum()
        contains_apostrophes = lyric_series.str.contains(r"'", regex=False).sum()

        vprint(verbose, f"Rows with bracketed tags like [Chorus]: {contains_brackets}")
        vprint(verbose, f"Rows with newline characters: {contains_newlines}")
        vprint(verbose, f"Rows with apostrophes: {contains_apostrophes}")


def extract_lyrics(
    df: pd.DataFrame,
    min_words: int = 20,
    min_chars: int = 200,
    max_chars: int = 2825,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Extract lyric texts from the Lyric column.

    Filtering steps:
    1. Remove missing lyrics
    2. Remove rows where Album == 'Unreleased Songs'
    3. Remove lyrics shorter than min_words
    4. Remove lyrics shorter than min_chars
    5. Remove lyrics longer than max_chars
    """
    working_df = df.dropna(subset=["Lyric"]).copy()
    working_df["Lyric"] = working_df["Lyric"].astype(str)

    before_count = len(working_df)

    # Remove unreleased songs
    unreleased_mask = (
        working_df["Album"]
        .fillna("")
        .str.strip()
        .str.lower()
        == "unreleased songs"
    )
    removed_unreleased = unreleased_mask.sum()
    working_df = working_df.loc[~unreleased_mask].copy()

    # Length columns
    working_df["word_count"] = working_df["Lyric"].str.split().str.len()
    working_df["char_count"] = working_df["Lyric"].str.len()

    # Remove short lyrics by word count
    short_word_mask = working_df["word_count"] < min_words
    removed_short_words = short_word_mask.sum()
    working_df = working_df.loc[~short_word_mask].copy()

    # Remove short lyrics by character count
    short_char_mask = working_df["char_count"] < min_chars
    removed_short_chars = short_char_mask.sum()
    working_df = working_df.loc[~short_char_mask].copy()

    # Remove long lyrics by character count
    long_char_mask = working_df["char_count"] > max_chars
    removed_long_chars = long_char_mask.sum()
    working_df = working_df.loc[~long_char_mask].copy()

    after_count = len(working_df)

    # Recompute after filtering to ensure stats reflect final corpus
    working_df["word_count"] = working_df["Lyric"].str.split().str.len()
    working_df["char_count"] = working_df["Lyric"].str.len()

    vprint(verbose, "\n--- FILTERING DECISIONS ---")
    vprint(verbose, f"Removed unreleased songs: {removed_unreleased}")
    vprint(verbose, f"Removed short lyrics (< {min_words} words): {removed_short_words}")
    vprint(verbose, f"Removed short lyrics (< {min_chars} chars): {removed_short_chars}")
    vprint(verbose, f"Removed long lyrics (> {max_chars} chars): {removed_long_chars}")
    vprint(verbose, f"Rows before filtering: {before_count}")
    vprint(verbose, f"Rows after filtering: {after_count}")
    vprint(verbose, f"Total removed: {before_count - after_count}")

    vprint(verbose, "\n--- POST-FILTER LENGTH STATS (CHARACTERS) ---")
    vprint(verbose, working_df["char_count"].describe())

    vprint(verbose, "\n--- POST-FILTER LENGTH STATS (WORDS) ---")
    vprint(verbose, working_df["word_count"].describe())

    # Explicit boundary checks
    if not working_df.empty:
        vprint(verbose, "\n--- FILTER BOUNDARY CHECKS ---")
        vprint(verbose, f"Minimum characters in final corpus: {working_df['char_count'].min()}")
        vprint(verbose, f"Maximum characters in final corpus: {working_df['char_count'].max()}")
        vprint(verbose, f"Minimum words in final corpus: {working_df['word_count'].min()}")
        vprint(verbose, f"Maximum words in final corpus: {working_df['word_count'].max()}")

    return working_df["Lyric"].tolist(), working_df


def inspect_filtered_dataset(filtered_df: pd.DataFrame, verbose: bool = True) -> None:
    """Print summary statistics for the final filtered dataset."""
    if filtered_df.empty:
        vprint(verbose, "\n--- FILTERED DATASET SUMMARY ---")
        vprint(verbose, "No rows left after filtering.")
        return

    vprint(verbose, "\n--- FILTERED DATASET SUMMARY ---")
    vprint(verbose, f"Final dataset shape: {filtered_df.shape}")
    vprint(verbose, f"Unique albums after filtering: {filtered_df['Album'].fillna('Unknown').nunique()}")

    if "Album" in filtered_df.columns:
        vprint(verbose, "\n--- TOP ALBUMS AFTER FILTERING ---")
        vprint(verbose, filtered_df["Album"].fillna("Unknown").value_counts().head(10))

    if "Year" in filtered_df.columns:
        vprint(verbose, "\n--- YEAR DISTRIBUTION AFTER FILTERING ---")
        vprint(verbose, filtered_df["Year"].value_counts().sort_index())

    if "Title" in filtered_df.columns:
        vprint(verbose, "\n--- SAMPLE TITLES AFTER FILTERING ---")
        vprint(verbose, filtered_df["Title"].head(10).tolist())


def inspect_lyrics_list(lyrics: list[str], verbose: bool = True) -> None:
    """Inspect extracted lyrics after filtering."""
    vprint(verbose, "\n--- EXTRACTED LYRICS AFTER FILTERING ---")
    vprint(verbose, f"Number of songs with lyrics: {len(lyrics)}")

    if not lyrics:
        vprint(verbose, "No lyrics found after filtering.")
        return

    total_characters = sum(len(text) for text in lyrics)
    total_words = sum(len(text.split()) for text in lyrics)

    vprint(verbose, "\n--- CORPUS SIZE ---")
    vprint(verbose, f"Total characters across lyrics: {total_characters}")
    vprint(verbose, f"Total words across lyrics: {total_words}")

    shortest = min(lyrics, key=len)
    longest = max(lyrics, key=len)

    vprint(verbose, "\n--- SHORTEST / LONGEST LYRIC PREVIEW AFTER FILTERING ---")
    vprint(verbose, f"Shortest lyric length: {len(shortest)} chars")
    vprint(verbose, shortest[:300])

    vprint(verbose, f"\nLongest lyric length: {len(longest)} chars")
    vprint(verbose, longest[:300])

    vprint(verbose, "\n--- 3 START-OF-LYRIC EXAMPLES ---")
    for i, lyric in enumerate(lyrics[:3], start=1):
        vprint(verbose, f"\nStart example {i}:")
        vprint(verbose, lyric[:500])

    vprint(verbose, "\n--- 3 END-OF-LYRIC EXAMPLES ---")
    for i, lyric in enumerate(lyrics[:3], start=1):
        vprint(verbose, f"\nEnd example {i}:")
        vprint(verbose, lyric[-500:])


# FUNCTIONS: PARSER
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect Taylor Swift lyric data before training an n-gram model."
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
        default=OUTPUT_PATH,
        help="Path where the trained model will later be saved.",
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

    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable verbose output. Default is True.",
    )

    return parser.parse_args()


# FUNCTIONS: MAIN
def main(
    input_path: Path,
    output_path: Path,
    min_words: int = 20,
    min_chars: int = 500,
    max_chars: int = 2825,
    verbose: bool = True,
) -> None:
    """Main inspection pipeline before later model training."""
    vprint(verbose, "\n=== TRAINING SCRIPT: DATA INSPECTION MODE ===")
    vprint(verbose, f"Input path: {input_path}")
    vprint(verbose, f"Planned output path: {output_path}")
    vprint(verbose, f"Minimum words per lyric: {min_words}")
    vprint(verbose, f"Minimum characters per lyric: {min_chars}")
    vprint(verbose, f"Maximum characters per lyric: {max_chars}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_data(input_path)
    inspect_data(df, verbose=verbose)

    lyrics, filtered_df = extract_lyrics(
        df,
        min_words=min_words,
        min_chars=min_chars,
        max_chars=max_chars,
        verbose=verbose,
    )

    inspect_filtered_dataset(filtered_df, verbose=verbose)
    inspect_lyrics_list(lyrics, verbose=verbose)

    vprint(verbose, "\n=== NEXT STEP ===")
    vprint(verbose, "Data loading and filtering works. Next we can add preprocessing and NgramModel training.")


# CALL MAIN
if __name__ == "__main__":
    args = parse_args()
    main(
        input_path=args.input,
        output_path=args.output,
        min_words=args.min_words,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        verbose=args.verbose,
    )