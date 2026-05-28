"""
linguistic_profile.py

Creates a yearly linguistic profile for one preprocessed arXiv dataset.

The script:
1. Calculates stylistic metrics per abstract
2. Aggregates metrics by year using mean and standard deviation
3. Saves the yearly linguistic profile in out/linguistic_profile/
4. Returns both the abstract-level dataframe and the yearly profile
"""

# IMPORTS
import re
from pathlib import Path

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords


# GLOBALS
try:
    STOPWORDS = set(stopwords.words("english"))

except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])


# FUNCTION: MAIN
def create_linguistic_profile(df, dataset_name, root_dir):
    """
    Creates and saves yearly linguistic profile for a preprocessed dataset.

    Returns:
        df: dataframe with abstract-level linguistic metrics
        profile_df: yearly aggregated linguistic profile
    """

    df = df.copy()

    df = add_word_and_sentence_length_metrics(df)
    df = add_mattr_metric(df)
    df = add_hapax_metric(df)
    df = add_stopword_ratio_metric(df)
    df = add_pos_metrics(df)

    profile_df = aggregate_linguistic_profile(df)

    save_linguistic_profile(profile_df, dataset_name, root_dir)

    print_linguistic_profile_preview(profile_df)

    return df, profile_df


# FUNCTION: SUPPORT
def add_word_and_sentence_length_metrics(df):
    """
    Adds word and sentence length metrics to each abstract.
    """

    df["mean_word_length"] = df["processed_abstract"].apply(
        calculate_mean_word_length
    )

    df["mean_sentence_length"] = df["processed_abstract"].apply(
        calculate_mean_sentence_length
    )

    return df


def add_mattr_metric(df):
    """
    Adds MATTR lexical diversity metric to each abstract.
    """

    df["mattr"] = df["processed_abstract"].apply(calculate_mattr)

    return df


def add_hapax_metric(df):
    """
    Adds hapax ratio metric to each abstract.
    """

    df["hapax_ratio"] = df["processed_abstract"].apply(
        calculate_hapax_ratio
    )

    return df


def add_stopword_ratio_metric(df):
    """
    Adds stopword ratio metric to each abstract.
    """

    df["stopword_ratio"] = df["processed_abstract"].apply(
        calculate_stopword_ratio
    )

    return df


def add_pos_metrics(df):
    """
    Adds POS-based stylistic metrics to each abstract.

    The selected POS ratios capture broad grammatical style:
    - noun_ratio: information density and nominal academic style
    - verb_ratio: process/action orientation
    - adjective_ratio: descriptive or evaluative modification
    """

    pos_metrics = df["processed_abstract"].apply(calculate_pos_ratios)

    df["noun_ratio"] = pos_metrics.apply(lambda x: x["noun_ratio"])
    df["verb_ratio"] = pos_metrics.apply(lambda x: x["verb_ratio"])
    df["adjective_ratio"] = pos_metrics.apply(
        lambda x: x["adjective_ratio"]
    )

    return df


def aggregate_linguistic_profile(df):
    """
    Aggregates abstract-level linguistic metrics by year.
    """

    profile_df = (
        df.groupby("year")
        .agg(
            mean_word_length=("mean_word_length", "mean"),
            sd_word_length=("mean_word_length", "std"),

            mean_sentence_length=("mean_sentence_length", "mean"),
            sd_sentence_length=("mean_sentence_length", "std"),

            mean_mattr=("mattr", "mean"),
            sd_mattr=("mattr", "std"),

            mean_hapax_ratio=("hapax_ratio", "mean"),
            sd_hapax_ratio=("hapax_ratio", "std"),

            mean_stopword_ratio=("stopword_ratio", "mean"),
            sd_stopword_ratio=("stopword_ratio", "std"),

            mean_noun_ratio=("noun_ratio", "mean"),
            sd_noun_ratio=("noun_ratio", "std"),

            mean_verb_ratio=("verb_ratio", "mean"),
            sd_verb_ratio=("verb_ratio", "std"),

            mean_adjective_ratio=("adjective_ratio", "mean"),
            sd_adjective_ratio=("adjective_ratio", "std"),
        )
        .reset_index()
    )

    profile_df = profile_df.round(3)

    return profile_df


def calculate_mean_word_length(text):
    """
    Calculates mean word length for one abstract.
    """

    words = get_words(text)

    if len(words) == 0:
        return 0

    word_lengths = [len(word) for word in words]

    return sum(word_lengths) / len(word_lengths)


def calculate_mean_sentence_length(text):
    """
    Calculates mean sentence length in words for one abstract.
    """

    sentence_lengths = get_sentence_lengths(text)

    if len(sentence_lengths) == 0:
        return 0

    return sum(sentence_lengths) / len(sentence_lengths)


def calculate_mattr(text, window_size=50):
    """
    Calculates Moving-Average Type-Token Ratio (MATTR).

    MATTR estimates lexical diversity while reducing sensitivity to text length.
    """

    words = get_words(text)

    if len(words) < window_size:
        return calculate_ttr(words)

    ttr_scores = []

    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        ttr_scores.append(calculate_ttr(window))

    return sum(ttr_scores) / len(ttr_scores)


def calculate_ttr(words):
    """
    Calculates Type-Token Ratio for a list of words.
    """

    if len(words) == 0:
        return 0

    unique_words = set(words)

    return len(unique_words) / len(words)


def calculate_hapax_ratio(text):
    """
    Calculates hapax ratio for one abstract.

    Hapax ratio measures the proportion of words
    that occur only once in a text.
    """

    words = get_words(text)

    if len(words) == 0:
        return 0

    word_frequencies = pd.Series(words).value_counts()

    hapax_count = (word_frequencies == 1).sum()

    return hapax_count / len(words)


def calculate_stopword_ratio(text):
    """
    Calculates stopword ratio for one abstract.

    Stopword ratio measures the proportion of function words
    relative to all words in a text.
    """

    words = get_words(text)

    if len(words) == 0:
        return 0

    stopword_count = sum(
        1 for word in words
        if word in STOPWORDS
    )

    return stopword_count / len(words)


def calculate_pos_ratios(text):
    """
    Calculates noun, verb, and adjective ratios for one abstract.
    """

    doc = NLP(str(text))

    tokens = [
        token for token in doc
        if token.is_alpha
    ]

    if len(tokens) == 0:
        return {
            "noun_ratio": 0,
            "verb_ratio": 0,
            "adjective_ratio": 0,
        }

    noun_count = sum(
        1 for token in tokens
        if token.pos_ in ["NOUN", "PROPN"]
    )

    verb_count = sum(
        1 for token in tokens
        if token.pos_ in ["VERB", "AUX"]
    )

    adjective_count = sum(
        1 for token in tokens
        if token.pos_ == "ADJ"
    )

    total_tokens = len(tokens)

    return {
        "noun_ratio": noun_count / total_tokens,
        "verb_ratio": verb_count / total_tokens,
        "adjective_ratio": adjective_count / total_tokens,
    }


def get_words(text):
    """
    Splits processed text into word tokens.
    """

    words = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b", str(text))

    return words


def get_sentence_lengths(text):
    """
    Splits processed text into sentences and returns sentence lengths in words.
    """

    sentences = re.split(r"[.!?]+", str(text))

    sentence_lengths = []

    for sentence in sentences:
        words = get_words(sentence)

        if len(words) > 0:
            sentence_lengths.append(len(words))

    return sentence_lengths


def save_linguistic_profile(profile_df, dataset_name, root_dir):
    """
    Saves linguistic profile CSV.
    """

    output_dir = Path(root_dir) / "out" / "linguistic_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_linguistic_profile.csv"

    profile_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("LINGUISTIC PROFILE SAVED")
    print("=" * 80)

    print(f"\nSaved to:\n{output_path}")


def print_linguistic_profile_preview(profile_df):
    """
    Prints preview of linguistic profile table.
    """

    print("\nPreview:\n")
    print(profile_df.to_string(index=False))