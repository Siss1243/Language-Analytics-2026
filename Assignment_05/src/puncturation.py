"""
punctuation_analysis.py

Analyzes punctuation use in processed abstracts as stylistic features.

The script:
1. Calculates punctuation counts per abstract
2. Calculates punctuation rates per 100 words per abstract
3. Aggregates punctuation metrics by year using mean and standard deviation
4. Saves the yearly punctuation profile in out/punctuation/
5. Returns both the abstract-level dataframe and the yearly profile
"""

from pathlib import Path

import pandas as pd


PUNCTUATION_DIR_NAME = "punctuation"


def create_punctuation_analysis(df, dataset_name, root_dir):
    """
    Creates yearly punctuation statistics for one dataset.

    Returns:
        df: dataframe with abstract-level punctuation features
        yearly_df: yearly aggregated punctuation profile
    """

    df = df.copy()

    df = add_punctuation_features(df)

    yearly_df = aggregate_punctuation_by_year(df)

    save_punctuation_profile(yearly_df, dataset_name, root_dir)

    print_punctuation_preview(yearly_df)

    return df, yearly_df


def add_punctuation_features(df):
    """
    Calculates punctuation counts and punctuation rates per 100 words.
    """

    for index, row in df.iterrows():
        text = str(row["processed_abstract"])
        word_count = row["processed_abstract_word_count"]

        features = {
            "comma": text.count(","),
            "period": text.count("."),
            "semicolon": text.count(";"),
            "colon": text.count(":"),
            "question_mark": text.count("?"),
            "exclamation_mark": text.count("!"),
            "parenthesis": text.count("(") + text.count(")"),
            "dash": text.count("-"),
        }

        for name, count in features.items():
            df.loc[index, f"{name}_count"] = count
            df.loc[index, f"{name}_rate"] = rate_per_100_words(
                count,
                word_count,
            )

    return df


def rate_per_100_words(count, word_count):
    """
    Normalizes punctuation count per 100 words.
    """

    if word_count == 0:
        return 0

    return (count / word_count) * 100


def aggregate_punctuation_by_year(df):
    """
    Aggregates punctuation features by year using mean and standard deviation.
    """

    punctuation_features = [
        "comma_rate",
        "period_rate",
        "semicolon_rate",
        "colon_rate",
        "question_mark_rate",
        "exclamation_mark_rate",
        "parenthesis_rate",
        "dash_rate",
    ]

    rows = []

    for year, group in df.groupby("year"):
        row = {
            "year": year,
            "texts": len(group),
        }

        for feature in punctuation_features:
            row[f"mean_{feature}"] = round(group[feature].mean(), 4)
            row[f"sd_{feature}"] = round(group[feature].std(), 4)

        rows.append(row)

    yearly_df = pd.DataFrame(rows)

    return yearly_df


def save_punctuation_profile(yearly_df, dataset_name, root_dir):
    """
    Saves yearly punctuation profile as CSV.
    """

    output_dir = Path(root_dir) / "out" / PUNCTUATION_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_punctuation_profile.csv"

    yearly_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("PUNCTUATION ANALYSIS SAVED")
    print("=" * 80)
    print(f"\nSaved to:\n{output_path}")


def print_punctuation_preview(yearly_df):
    """
    Prints preview of yearly punctuation profile.
    """

    print("\nPreview:\n")
    print(yearly_df.to_string(index=False))