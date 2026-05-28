"""
compare_models.py

Compare topic models based on topic coherence and topic-label differences.
"""

# IMPORTS
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


# PATH HANDLING
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
RESULTS_DIR = OUT_DIR / "results"


def main():
    print("Running compare_models.py...")

    result_dirs = get_result_dirs()
    summaries = []

    for result_dir in result_dirs:
        summary = summarize_model(result_dir)

        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise FileNotFoundError("No valid model result folders found.")

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values("n_topics")

    output_path = RESULTS_DIR / "model_comparison.csv"
    summary_df.to_csv(output_path, index=False)

    plot_model_comparison(summary_df)

    print("\nMODEL COMPARISON")
    print("----------------")
    print(summary_df)

    print(f"\nSaved comparison to: {output_path}")

    extract_selected_examples()


def get_result_dirs():
    """
    Find all model result folders in out/results.
    """

    return [
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir()
    ]


def summarize_model(result_dir):
    """
    Summarize one model run using saved analysis outputs.
    """

    coherence_path = result_dir / "topic_coherence.csv"
    differences_path = result_dir / "topic_differences.csv"
    topics_path = result_dir / "topics.csv"

    if not coherence_path.exists():
        print(f"Skipping {result_dir.name}: no topic_coherence.csv")
        return None

    if not differences_path.exists():
        print(f"Skipping {result_dir.name}: no topic_differences.csv")
        return None

    coherence_df = pd.read_csv(coherence_path)
    differences_df = pd.read_csv(differences_path)

    n_topics = infer_n_topics(result_dir.name, coherence_df)

    avg_coherence = coherence_df["coherence_npmi"].mean()
    sd_coherence = coherence_df["coherence_npmi"].std()
    min_coherence = coherence_df["coherence_npmi"].min()
    max_coherence = coherence_df["coherence_npmi"].max()

    avg_abs_label_difference = differences_df["abs_difference"].mean()
    max_abs_label_difference = differences_df["abs_difference"].max()

    strongest_topic = differences_df.iloc[0]["topic"]
    strongest_difference = differences_df.iloc[0]["story_minus_no_story"]

    return {
        "model": result_dir.name,
        "n_topics": n_topics,
        "avg_coherence_npmi": round(avg_coherence, 4),
        "sd_coherence_npmi": round(sd_coherence, 4),
        "min_coherence_npmi": round(min_coherence, 4),
        "max_coherence_npmi": round(max_coherence, 4),
        "avg_abs_label_difference": round(avg_abs_label_difference, 4),
        "max_abs_label_difference": round(max_abs_label_difference, 4),
        "strongest_topic": strongest_topic,
        "strongest_story_minus_no_story": round(strongest_difference, 4),
    }

def extract_selected_examples():
    """
    Extract four examples for qualitative analysis:
    - story and no_story examples from the most story-associated topic
    - story and no_story examples from the most no_story-associated topic

    Uses the selected 10-topic model.
    """

    k10_dirs = [
        path for path in RESULTS_DIR.iterdir()
        if path.is_dir() and "k10" in path.name
    ]

    if not k10_dirs:
        raise FileNotFoundError("No k10 result folder found.")

    result_dir = sorted(k10_dirs)[-1]

    examples_path = result_dir / "topic_examples.csv"
    differences_path = result_dir / "topic_differences.csv"

    examples_df = pd.read_csv(examples_path)
    differences_df = pd.read_csv(differences_path)

    story_topic = (
        differences_df
        .sort_values("story_minus_no_story", ascending=False)
        .iloc[0]["topic"]
    )

    no_story_topic = (
        differences_df
        .sort_values("story_minus_no_story", ascending=True)
        .iloc[0]["topic"]
    )

    selected_rows = []

    for topic in [story_topic, no_story_topic]:
        for label in ["story", "no_story"]:
            subset = examples_df[
                (examples_df["topic"] == topic)
                & (examples_df["label"] == label)
            ]

            if subset.empty:
                continue

            best_example = (
                subset
                .sort_values("score", ascending=False)
                .head(1)
            )

            selected_rows.append(best_example)

    selected_df = pd.concat(selected_rows, ignore_index=True)

    output_path = result_dir / "selected_qualitative_examples.csv"
    selected_df.to_csv(output_path, index=False)

    print("\nSELECTED QUALITATIVE EXAMPLES")
    print("-----------------------------")
    print(selected_df[["topic", "label", "score", "text"]])

    print(f"\nSaved selected examples to: {output_path}")

def infer_n_topics(model_name, coherence_df):
    """
    Infer number of topics from folder name, fallback to row count.
    """

    match = re.search(r"k(\d+)", model_name)

    if match:
        return int(match.group(1))

    return len(coherence_df)


def plot_model_comparison(summary_df):
    """
    Plot average coherence by number of topics.
    """

    plt.figure(figsize=(7, 5))

    plt.plot(
        summary_df["n_topics"],
        summary_df["avg_coherence_npmi"],
        marker="o",
    )

    plt.xlabel("Number of topics")
    plt.ylabel("Average NPMI coherence")
    plt.title("Average topic coherence across models")
    plt.xticks(summary_df["n_topics"])
    plt.tight_layout()

    plot_path = RESULTS_DIR / "model_coherence_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()