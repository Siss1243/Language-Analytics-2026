"""
analyze_data.py

Loads one sampled arXiv dataset and runs the full stylistic analysis pipeline.

The script:
1. Loads one sampled category dataset from the in/ folder
2. Prints descriptive information about the raw dataset
3. Applies conservative stylistic preprocessing
4. Creates descriptive statistics
5. Creates linguistic profile metrics
6. Creates punctuation profile metrics
7. Creates distance-to-centroid analysis
8. Saves abstract-level style profiles for later plotting
9. Creates n-gram analyses
"""

# IMPORTS
import argparse
import re
from pathlib import Path

import pandas as pd

from linguistic_profile import create_linguistic_profile
from puncturation import create_punctuation_analysis
from centroid_analysis import create_centroid_analysis
from n_gram import create_ngram_analysis


# PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent

IN_DIR = ROOT_DIR / "in"

OUT_DIR = ROOT_DIR / "out"
DESCRIPTIVE_DIR = OUT_DIR / "descriptive"
ABSTRACT_PROFILE_DIR = OUT_DIR / "abstract_profiles"

DATASET_PATHS = {
    "cs.CL": IN_DIR / "cs_cl_2016_2026.csv",
    "math-ph": IN_DIR / "math_ph_2016_2026.csv",
    "quant-ph": IN_DIR / "quant_ph_2016_2026.csv",
}


# FUNCTION: MAIN
def main():
    args = parse_args()

    df = load_dataset(DATASET_PATHS[args.dataset])

    print_descriptive_info(df, args.dataset)

    df = preprocess_dataset(df)

    print_preprocessing_info(df)

    print_preprocessing_example(df)

    create_descriptive_table(df, args.dataset)

    df, linguistic_profile_df = create_linguistic_profile(
        df,
        args.dataset,
        ROOT_DIR,
    )

    df, punctuation_profile_df = create_punctuation_analysis(
        df,
        args.dataset,
        ROOT_DIR,
    )

    df, centroid_df = create_centroid_analysis(
        df,
        args.dataset,
        ROOT_DIR,
    )

    save_abstract_style_profile(df, args.dataset)

    #create_ngram_analysis(df,args.dataset, ROOT_DIR,)


# FUNCTION: SUPPORT
def load_dataset(input_file):
    """
    Loads sampled arXiv dataset.
    """

    df = pd.read_csv(input_file)

    return df


def print_descriptive_info(df, dataset_name):
    """
    Prints basic descriptive information about the raw dataset.
    """

    print("\n" + "=" * 80)
    print(f"DATASET OVERVIEW: {dataset_name}")
    print("=" * 80)

    print(f"\nNumber of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nPapers per year:")
    print(df["year"].value_counts().sort_index().to_string())

    print("\nRaw abstract length overview:")
    abstract_lengths = df["abstract"].astype(str).str.split().str.len()

    print(abstract_lengths.describe().to_string())


def preprocess_dataset(df):
    """
    Applies preprocessing to all abstracts.
    """

    df = df.copy()

    df["processed_abstract"] = df["abstract"].apply(preprocess_abstract)

    df["processed_abstract_word_count"] = (
        df["processed_abstract"]
        .astype(str)
        .str.split()
        .str.len()
    )

    return df


def preprocess_abstract(text):
    """
    Conservatively preprocesses abstracts for stylistic similarity analysis.

    The goal is to preserve stylistic signals while removing obvious technical noise.
    """

    text = str(text)

    # remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove inline LaTeX/math expressions
    text = re.sub(r"\$.*?\$", " ", text)

    # normalize different dash types, but keep dash signal
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # normalize quotation marks
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # lowercase
    text = text.lower()

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def print_preprocessing_info(df):
    """
    Prints descriptive information about preprocessing effects.
    """

    print("\n" + "=" * 80)
    print("PREPROCESSING OVERVIEW")
    print("=" * 80)

    length_comparison = df[
        [
            "abstract_word_count",
            "processed_abstract_word_count",
        ]
    ].describe()

    print("\nWord count before and after preprocessing:")
    print(length_comparison.to_string())

    df["removed_words_after_preprocessing"] = (
        df["abstract_word_count"] - df["processed_abstract_word_count"]
    )

    print("\nRemoved word count overview:")
    print(df["removed_words_after_preprocessing"].describe().to_string())


def print_preprocessing_example(df):
    """
    Prints the same random abstract before and after preprocessing.
    """

    example = df.sample(n=1, random_state=42).iloc[0]

    print("\n" + "=" * 80)
    print("RAW VS PREPROCESSED ABSTRACT EXAMPLE")
    print("=" * 80)

    print(f"\nID: {example['id']}")
    print(f"YEAR: {example['year']}")
    print(f"TITLE: {example['title']}")

    print("\nRAW ABSTRACT:\n")
    print(example["abstract"])

    print("\nPROCESSED ABSTRACT:\n")
    print(example["processed_abstract"])

    print("\n" + "-" * 80)


def create_descriptive_table(df, dataset_name):
    """
    Creates yearly descriptive statistics table for processed abstracts.
    """

    descriptive_rows = []

    for year, group in df.groupby("year"):

        word_counts = group["processed_abstract_word_count"]

        descriptive_rows.append(
            {
                "year": year,
                "texts": len(group),
                "total_words": word_counts.sum(),
                "mean_words": round(word_counts.mean(), 2),
                "words_sd": round(word_counts.std(), 2),
                "word_range": (
                    f"{word_counts.min()}-{word_counts.max()}"
                ),
            }
        )

    descriptive_df = pd.DataFrame(descriptive_rows)

    DESCRIPTIVE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        DESCRIPTIVE_DIR /
        f"{dataset_name}_descriptive_statistics.csv"
    )

    descriptive_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("DESCRIPTIVE TABLE SAVED")
    print("=" * 80)

    print(f"\nSaved to:\n{output_path}")

    print("\nPreview:\n")
    print(descriptive_df.to_string(index=False))


def save_abstract_style_profile(df, dataset_name):
    """
    Saves abstract-level stylistic profile for later plotting.

    This file makes it possible to create violinplots and other distribution
    plots without rerunning the full NLP pipeline.
    """

    ABSTRACT_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    profile_columns = [
        "id",
        "year",
        "mean_sentence_length",
        "mean_word_length",
        "stopword_ratio",
        "mattr",
        "hapax_ratio",
        "noun_ratio",
        "verb_ratio",
        "adjective_ratio",
        "dash_rate",
        "comma_rate",
        "semicolon_rate",
        "distance_to_centroid",
    ]

    available_columns = [
        column for column in profile_columns
        if column in df.columns
    ]

    output_path = (
        ABSTRACT_PROFILE_DIR /
        f"{dataset_name}_abstract_style_profile.csv"
    )

    df[available_columns].to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("ABSTRACT STYLE PROFILE SAVED")
    print("=" * 80)
    print(f"\nSaved to:\n{output_path}")

    print("\nColumns saved:\n")
    print(available_columns)


# FUNCTION: PARSER
def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run stylistic convergence analysis for one arXiv dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="quant-ph",
        choices=list(DATASET_PATHS.keys()),
        help="Dataset to analyze. Default: quant-ph",
    )

    return parser.parse_args()


# CALL FOR MAIN
if __name__ == "__main__":
    main()