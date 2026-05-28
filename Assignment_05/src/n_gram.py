"""
ngram_analysis.py

Creates corpus-level n-gram analyses for one preprocessed arXiv dataset.

The script:
1. Extracts top unigrams, bigrams, and trigrams per year
2. Creates both unfiltered and content-filtered n-gram versions
3. Calculates overlap between yearly top n-gram lists
4. Saves shared n-gram examples for qualitative inspection

The final outputs are saved in out/ngrams/.
"""

# IMPORTS
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords


# GLOBALS
NGRAM_SIZES = [1, 2, 3]
TOP_N = 50

STOPWORDS = set(stopwords.words("english"))

NGRAM_FILTERS = [
    "unfiltered",
    "content_filtered",
]


# FUNCTION: MAIN
def create_ngram_analysis(df, dataset_name, root_dir):
    """
    Creates and saves n-gram analysis outputs.
    """

    df = df.copy()

    top_ngrams_df = create_top_ngrams_table(df)

    overlap_df = create_ngram_overlap_table(top_ngrams_df)

    shared_examples_df = create_shared_ngram_examples(top_ngrams_df)

    save_ngram_outputs(
        top_ngrams_df,
        overlap_df,
        shared_examples_df,
        dataset_name,
        root_dir,
    )

    print_ngram_preview(
        top_ngrams_df,
        overlap_df,
        shared_examples_df,
    )


# FUNCTION: SUPPORT
def create_top_ngrams_table(df):
    """
    Creates table of top n-grams per year.

    Two versions are created:
    - unfiltered: all n-grams
    - content_filtered: only n-grams containing at least one non-stopword

    For unigrams, the content-filtered version corresponds to top words
    with stopwords removed.
    """

    rows = []

    for year, group in df.groupby("year"):

        tokens = get_year_tokens(group["processed_abstract"])

        for ngram_size in NGRAM_SIZES:

            for ngram_filter in NGRAM_FILTERS:

                ngram_counts = count_ngrams(
                    tokens,
                    ngram_size,
                    ngram_filter,
                )

                for ngram, frequency in ngram_counts.most_common(TOP_N):

                    rows.append(
                        {
                            "year": year,
                            "ngram_size": ngram_size,
                            "ngram_filter": ngram_filter,
                            "ngram": " ".join(ngram),
                            "frequency": frequency,
                        }
                    )

    top_ngrams_df = pd.DataFrame(rows)

    return top_ngrams_df


def create_ngram_overlap_table(top_ngrams_df):
    """
    Calculates overlap between yearly top n-gram lists.

    Overlap is calculated separately for each n-gram size and filter type.
    """

    rows = []

    for ngram_size in NGRAM_SIZES:

        for ngram_filter in NGRAM_FILTERS:

            size_df = top_ngrams_df[
                (top_ngrams_df["ngram_size"] == ngram_size)
                & (top_ngrams_df["ngram_filter"] == ngram_filter)
            ]

            years = sorted(size_df["year"].unique())

            for year_a, year_b in combinations(years, 2):

                ngrams_a = set(
                    size_df[size_df["year"] == year_a]["ngram"]
                )

                ngrams_b = set(
                    size_df[size_df["year"] == year_b]["ngram"]
                )

                shared_ngrams = ngrams_a.intersection(ngrams_b)
                union_ngrams = ngrams_a.union(ngrams_b)

                rows.append(
                    {
                        "year_a": year_a,
                        "year_b": year_b,
                        "ngram_size": ngram_size,
                        "ngram_filter": ngram_filter,
                        "top_n": TOP_N,
                        "shared_count": len(shared_ngrams),
                        "jaccard_similarity": (
                            len(shared_ngrams) / len(union_ngrams)
                            if len(union_ngrams) > 0
                            else 0
                        ),
                    }
                )

    overlap_df = pd.DataFrame(rows)

    overlap_df["jaccard_similarity"] = (
        overlap_df["jaccard_similarity"].round(3)
    )

    return overlap_df


def create_shared_ngram_examples(top_ngrams_df):
    """
    Saves examples of n-grams shared between pairs of years.

    This is useful for qualitative interpretation of repeated discourse patterns.
    """

    rows = []

    for ngram_size in NGRAM_SIZES:

        for ngram_filter in NGRAM_FILTERS:

            size_df = top_ngrams_df[
                (top_ngrams_df["ngram_size"] == ngram_size)
                & (top_ngrams_df["ngram_filter"] == ngram_filter)
            ]

            years = sorted(size_df["year"].unique())

            for year_a, year_b in combinations(years, 2):

                year_a_df = size_df[size_df["year"] == year_a]
                year_b_df = size_df[size_df["year"] == year_b]

                ngrams_a = set(year_a_df["ngram"])
                ngrams_b = set(year_b_df["ngram"])

                shared_ngrams = sorted(
                    ngrams_a.intersection(ngrams_b)
                )

                for ngram in shared_ngrams:

                    frequency_a = year_a_df[
                        year_a_df["ngram"] == ngram
                    ]["frequency"].iloc[0]

                    frequency_b = year_b_df[
                        year_b_df["ngram"] == ngram
                    ]["frequency"].iloc[0]

                    rows.append(
                        {
                            "year_a": year_a,
                            "year_b": year_b,
                            "ngram_size": ngram_size,
                            "ngram_filter": ngram_filter,
                            "ngram": ngram,
                            "frequency_year_a": frequency_a,
                            "frequency_year_b": frequency_b,
                        }
                    )

    shared_examples_df = pd.DataFrame(rows)

    return shared_examples_df


def get_year_tokens(texts):
    """
    Combines all abstracts from one year and tokenizes them.
    """

    combined_text = " ".join(texts.astype(str))

    tokens = get_ngram_tokens(combined_text)

    return tokens


def get_ngram_tokens(text):
    """
    Tokenizes text for n-gram analysis.

    The tokenization is conservative:
    - keeps stopwords
    - keeps stylistic function words
    - removes standalone punctuation
    - removes one-character tokens to reduce equation noise
    """

    raw_tokens = re.findall(
        r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b",
        str(text).lower(),
    )

    tokens = [
        token for token in raw_tokens
        if len(token) >= 2
    ]

    return tokens


def count_ngrams(tokens, ngram_size, ngram_filter):
    """
    Counts n-grams of a given size and filter type.
    """

    ngrams = zip(
        *[
            tokens[i:]
            for i in range(ngram_size)
        ]
    )

    if ngram_filter == "content_filtered":
        ngrams = [
            ngram for ngram in ngrams
            if contains_non_stopword(ngram)
        ]

    ngram_counts = Counter(ngrams)

    return ngram_counts


def contains_non_stopword(ngram):
    """
    Checks whether an n-gram contains at least one non-stopword.
    """

    return any(
        token not in STOPWORDS
        for token in ngram
    )


def save_ngram_outputs(
    top_ngrams_df,
    overlap_df,
    shared_examples_df,
    dataset_name,
    root_dir,
):
    """
    Saves n-gram output tables.
    """

    output_dir = Path(root_dir) / "out" / "ngrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    top_ngrams_path = output_dir / f"{dataset_name}_top_ngrams.csv"
    overlap_path = output_dir / f"{dataset_name}_ngram_overlap.csv"
    shared_examples_path = (
        output_dir / f"{dataset_name}_shared_ngrams.csv"
    )

    top_ngrams_df.to_csv(top_ngrams_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)
    shared_examples_df.to_csv(shared_examples_path, index=False)

    print("\n" + "=" * 80)
    print("N-GRAM ANALYSIS SAVED")
    print("=" * 80)

    print(f"\nTop n-grams saved to:\n{top_ngrams_path}")
    print(f"\nN-gram overlap saved to:\n{overlap_path}")
    print(f"\nShared n-gram examples saved to:\n{shared_examples_path}")


def print_ngram_preview(
    top_ngrams_df,
    overlap_df,
    shared_examples_df,
):
    """
    Prints preview of n-gram output tables.
    """

    print("\nTop n-grams preview:\n")
    print(top_ngrams_df.head(15).to_string(index=False))

    print("\nOverlap preview:\n")
    print(overlap_df.head(15).to_string(index=False))

    print("\nShared n-gram examples preview:\n")
    print(shared_examples_df.head(15).to_string(index=False))