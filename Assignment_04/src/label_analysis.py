"""
label_analysis.py

Analyze how KeyNMF topics relate to story/no_story labels.
"""

# IMPORTS
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import math
from itertools import combinations
import numpy as np

import joblib
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

lemmatizer = WordNetLemmatizer()


# PATH HANDLING
ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "in"
OUT_DIR = ROOT / "out"

PROCESSED_DATA_DIR = IN_DIR / "processed"
MODEL_DIR = OUT_DIR / "models"
RESULTS_DIR = OUT_DIR / "results"


# FUNCTIONS: MAIN

def main(args):
    print("Running label_analysis.py...")

    ensure_dir(RESULTS_DIR)

    model_path = resolve_latest_file(MODEL_DIR, "*.joblib", args.model_path)
    data_path = resolve_latest_file(PROCESSED_DATA_DIR, "*.csv", args.data_path)

    model = load_model(model_path)

    if hasattr(model, "model") and hasattr(model.model, "components"):
        model.model.components = model.model.components.astype("float64")

    df = load_data(data_path)

    topics = extract_topics(model, n_words=args.n_words)

    topic_coherence = calculate_topic_coherence(
        topics=topics,
        texts=df["processed_text"],)

    document_topics = get_document_topic_scores(model, df)
    topic_by_label = analyze_topics_by_label(document_topics)
    topic_differences = calculate_label_differences(topic_by_label)
    topic_examples = extract_topic_examples(
        document_topics=document_topics,
        n_examples=args.n_examples,)
    
    topics = topics.merge(topic_coherence, on="topic_id", how="left")

    run_id = model_path.stem

    save_outputs(
    topics=topics,
    document_topics=document_topics,
    topic_by_label=topic_by_label,
    topic_differences=topic_differences,
    topic_examples=topic_examples,
    topic_coherence=topic_coherence,
    run_id=run_id,
)

    print("\nAnalysis complete.")


# FUNCTIONS: SUPPORT

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_latest_file(directory, pattern, explicit_path=None):
    if explicit_path is not None:
        return Path(explicit_path)

    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found in {directory} matching {pattern}")

    return files[-1]


def load_model(path):
    model = joblib.load(path)
    print(f"Loaded model from: {path}")
    return model


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded processed data from: {path}")
    print("Shape:", df.shape)
    return df


def normalize_topic_word(word):
    """
    Lemmatize topic word for cleaner display.
    """

    word = word.lower().strip()

    # noun lemma first
    lemma = lemmatizer.lemmatize(word, pos="n")

    # then verb lemma
    lemma = lemmatizer.lemmatize(lemma, pos="v")

    return lemma

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_topic_words(words):
    """
    Clean and lemmatize extracted topic words.
    """

    custom_stopwords = {
        "just", "really", "like", "don",
        "ve", "ll", "tl", "dr"
    }

    stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)

    cleaned = []

    for word in words:

        word = normalize_topic_word(word)

        if word in stopwords:
            continue

        if len(word) <= 2:
            continue

        cleaned.append(word)

    # remove duplicates preserve order
    seen = set()
    final_words = []

    for word in cleaned:
        if word not in seen:
            final_words.append(word)
            seen.add(word)

    return final_words



def extract_topics(model, n_words=10):
    """
    Extract cleaned top words for each topic.
    """

    rows = []

    raw_topics = model.get_top_words(25)   # hent flere først

    for topic_id, words in enumerate(raw_topics):

        words = clean_topic_words(words)

        words = words[:n_words]

        rows.append({
            "topic_id": topic_id,
            "top_words": ", ".join(words),
        })

    topics_df = pd.DataFrame(rows)

    print("\nTOPICS")
    print("------")
    print(topics_df)

    return topics_df


def get_document_topic_scores(model, df):
    """
    Get document-topic scores and attach them to the dataset.
    """

    topic_matrix = model.transform(df["processed_text"].tolist())

    # fix dtype mismatch from KeyNMF / sklearn
    topic_matrix = np.asarray(topic_matrix, dtype=np.float32)

    topic_cols = [f"topic_{i}" for i in range(topic_matrix.shape[1])]
    topic_df = pd.DataFrame(topic_matrix, columns=topic_cols)

    document_topics = pd.concat(
        [df.reset_index(drop=True), topic_df.reset_index(drop=True)],
        axis=1,
    )

    document_topics["dominant_topic"] = topic_df.idxmax(axis=1)
    document_topics["dominant_topic_score"] = topic_df.max(axis=1)

    return document_topics


def analyze_topics_by_label(document_topics):
    """
    Calculate mean topic prevalence by label.
    """

    topic_cols = [
        col for col in document_topics.columns
        if col.startswith("topic_")
    ]

    topic_by_label = (
        document_topics
        .groupby("label")[topic_cols]
        .mean()
        .reset_index()
    )

    topic_by_label_long = topic_by_label.melt(
        id_vars="label",
        var_name="topic",
        value_name="mean_topic_score",
    )

    return topic_by_label_long

def calculate_label_differences(topic_by_label):
    """
    Calculate difference in topic prevalence between story and no_story.
    """

    wide_df = topic_by_label.pivot(
        index="topic",
        columns="label",
        values="mean_topic_score",
    ).reset_index()

    wide_df["story_minus_no_story"] = (
        wide_df["story"] - wide_df["no_story"]
    )

    wide_df["abs_difference"] = wide_df["story_minus_no_story"].abs()

    wide_df = wide_df.sort_values(
        "abs_difference",
        ascending=False
    )

    return wide_df


def calculate_topic_coherence(topics, texts):
    """
    Calculate simple NPMI topic coherence based on document co-occurrence.

    Higher values indicate that topic words tend to appear together
    in the same documents.
    """

    tokenized_docs = [
        set(text.split())
        for text in texts
        if isinstance(text, str) and text.strip()
    ]

    n_docs = len(tokenized_docs)

    rows = []

    for _, row in topics.iterrows():
        topic_id = row["topic_id"]
        words = row["top_words"].split(", ")

        word_pairs = list(combinations(words, 2))
        pair_scores = []

        for word_a, word_b in word_pairs:
            score = calculate_npmi(word_a, word_b, tokenized_docs, n_docs)

            if score is not None:
                pair_scores.append(score)

        coherence = (
            sum(pair_scores) / len(pair_scores)
            if pair_scores
            else None
        )

        rows.append({
            "topic_id": topic_id,
            "n_word_pairs": len(pair_scores),
            "coherence_npmi": coherence,
        })

    coherence_df = pd.DataFrame(rows)

    print("\nTOPIC COHERENCE")
    print("----------------")
    print(coherence_df)

    return coherence_df


def calculate_npmi(word_a, word_b, tokenized_docs, n_docs):
    """
    Calculate normalized pointwise mutual information for a word pair.
    """

    doc_count_a = sum(word_a in doc for doc in tokenized_docs)
    doc_count_b = sum(word_b in doc for doc in tokenized_docs)
    doc_count_ab = sum(
        word_a in doc and word_b in doc
        for doc in tokenized_docs
    )

    if doc_count_a == 0 or doc_count_b == 0 or doc_count_ab == 0:
        return None

    p_a = doc_count_a / n_docs
    p_b = doc_count_b / n_docs
    p_ab = doc_count_ab / n_docs

    pmi = math.log(p_ab / (p_a * p_b))
    npmi = pmi / (-math.log(p_ab))

    return npmi

def extract_topic_examples(document_topics, n_examples=3):
    """
    Extract top-scoring example documents for each topic and each label.
    """

    topic_cols = [
        col for col in document_topics.columns
        if col.startswith("topic_")
    ]

    rows = []

    for topic_col in topic_cols:
        for label in ["story", "no_story"]:
            subset = document_topics[document_topics["label"] == label]

            top_docs = (
                subset
                .sort_values(topic_col, ascending=False)
                .head(n_examples)
            )

            for _, row in top_docs.iterrows():
                rows.append({
                    "topic": topic_col,
                    "label": label,
                    "score": row[topic_col],
                    "text": row["text"],
                    "processed_text": row["processed_text"],
                })

    return pd.DataFrame(rows)


def save_outputs(
    topics,
    document_topics,
    topic_by_label,
    topic_differences,
    topic_examples,
    topic_coherence,
    run_id,):
    """
    Save outputs in a model-specific results folder.
    """

    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    topics.to_csv(run_dir / "topics.csv", index=False)
    document_topics.to_csv(run_dir / "document_topics.csv", index=False)
    topic_by_label.to_csv(run_dir / "topic_by_label.csv", index=False)
    topic_examples.to_csv(run_dir / "topic_examples.csv", index=False)
    topic_differences.to_csv(run_dir / "topic_differences.csv", index=False)
    topic_coherence.to_csv(run_dir / "topic_coherence.csv", index=False)

    plot_topic_prevalence_by_label(topic_by_label, run_dir)
    plot_topic_distribution_boxplot(document_topics, run_dir)

    print("\nSaved analysis outputs to:")
    print(run_dir)

def plot_topic_prevalence_by_label(topic_by_label, run_dir):
    """
    Create barplot of topic prevalence by label.
    """

    pivot_df = topic_by_label.pivot(
        index="topic",
        columns="label",
        values="mean_topic_score",
    )

    pivot_df.plot(kind="bar", figsize=(12, 6))

    topic_labels = [
        "General discourse",
        "Gaming & technology",
        "Business & economy",
        "Education & career",
        "Mobile technology & apps",
        "Consumer discussion",
        "Movement & physical events",
        "Reading & writing",
        "Health & body",
        "Shopping & pricing",
    ]

    plt.xticks(
        ticks=range(len(topic_labels)),
        labels=topic_labels,
        rotation=45,
        ha="right"
    )

    plt.title("Mean Topic Prevalence by Narrativity Label")
    plt.xlabel("Topic")
    plt.ylabel("Mean topic score")
    plt.tight_layout()

    plot_path = run_dir / "topic_prevalence_by_label.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {plot_path}")

def plot_topic_distribution_boxplot(document_topics, run_dir):
    """
    Create boxplots showing topic score distributions
    across story and no_story labels.
    """

    topic_cols = [
        col for col in document_topics.columns
        if col.startswith("topic_")
    ]

    # reshape to long format
    long_df = document_topics.melt(
        id_vars="label",
        value_vars=topic_cols,
        var_name="topic",
        value_name="topic_score",
    )

    # nicer topic labels
    topic_labels = {
        "topic_0": "General discourse",
        "topic_1": "Gaming & technology",
        "topic_2": "Business & economy",
        "topic_3": "Education & career",
        "topic_4": "Mobile technology & apps",
        "topic_5": "Consumer discussion",
        "topic_6": "Movement & physical events",
        "topic_7": "Reading & writing",
        "topic_8": "Health & body",
        "topic_9": "Shopping & pricing",
    }

    long_df["topic_label"] = long_df["topic"].map(topic_labels)

    fig, ax = plt.subplots(figsize=(14, 7))

    # create grouped boxplots
    long_df.boxplot(
        column="topic_score",
        by=["topic_label", "label"],
        ax=ax,
        grid=False,
        rot=45,
    )

    plt.suptitle("")  # remove automatic pandas title
    plt.title("Topic Score Distribution by Narrativity Label")
    plt.xlabel("Topic and Label")
    plt.ylabel("Topic score")
    plt.tight_layout()

    plot_path = run_dir / "topic_distribution_boxplot.png"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Boxplot saved to: {plot_path}")

# FUNCTIONS: PARSER

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze KeyNMF topics by StorySeeker labels."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to fitted KeyNMF model. If omitted, newest model is used.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to processed data. If omitted, newest processed data is used.",
    )

    parser.add_argument(
        "--n-words",
        type=int,
        default=10,
        help="Number of top words to extract per topic.",
    )

    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of high-scoring example documents per topic.",
    )

    return parser.parse_args()


# CALL FOR MAIN
if __name__ == "__main__":
    args = parse_args()
    main(args)