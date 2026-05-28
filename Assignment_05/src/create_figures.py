"""
create_figures.py

Creates exploratory plots from stored analysis outputs.

This script does NOT rerun the NLP pipeline.
It reads saved CSV files from out/ and saves plots in out/data_plots/.
"""

# IMPORTS
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GLOBAL PLOT STYLE
plt.rcParams.update({
    "font.size": 18,          # base font size
    "axes.titlesize": 21,     # plot titles
    "axes.labelsize": 18,     # axis labels
    "xtick.labelsize": 16,    # x-axis ticks
    "ytick.labelsize": 16,    # y-axis ticks
    "legend.fontsize": 16,    # legend text
    "legend.title_fontsize": 17,})


# PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "out"
PLOT_DIR = OUT_DIR / "data_plots"

ABSTRACT_PROFILE_DIR = OUT_DIR / "abstract_profiles"
LINGUISTIC_PROFILE_DIR = OUT_DIR / "linguistic_profile"
PUNCTUATION_DIR = OUT_DIR / "punctuation"
CENTROID_DIR = OUT_DIR / "centroid"
NGRAM_DIR = OUT_DIR / "ngrams"


# FEATURE GROUPS
PUNCTUATION_MEAN_COLUMNS = [
    "mean_comma_rate",
    "mean_period_rate",
    "mean_semicolon_rate",
    "mean_colon_rate",
    "mean_question_mark_rate",
    "mean_exclamation_mark_rate",
    "mean_parenthesis_rate",
    "mean_dash_rate",
]


POS_MEAN_COLUMNS = [
    "mean_noun_ratio",
    "mean_verb_ratio",
    "mean_adjective_ratio",
]


LEXICAL_STRUCTURAL_MEAN_COLUMNS = [
    "mean_sentence_length",
    "mean_word_length",
    "mean_stopword_ratio",
    "mean_mattr",
    "mean_hapax_ratio",
]

STRUCTURAL_SD_COLUMNS = [
    "sd_sentence_length",
    "sd_word_length",
]

LEXICAL_SD_COLUMNS = [
    "sd_mattr",
    "sd_hapax_ratio",
    "sd_stopword_ratio",
]

GRAMMATICAL_SD_COLUMNS = [
    "sd_noun_ratio",
    "sd_verb_ratio",
    "sd_adjective_ratio",
]

PUNCTUATION_SD_COLUMNS = [
    "sd_comma_rate",
    "sd_colon_rate",
    "sd_dash_rate",
]

VIOLIN_COLUMNS = [
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


# MAIN
def main():
    args = parse_args()
    dataset = args.dataset

    create_plot_directories()

    abstract_df = read_csv(
        ABSTRACT_PROFILE_DIR / f"{dataset}_abstract_style_profile.csv"
    )

    linguistic_df = read_csv(
        LINGUISTIC_PROFILE_DIR / f"{dataset}_linguistic_profile.csv"
    )

    punctuation_df = read_csv(
        PUNCTUATION_DIR / f"{dataset}_punctuation_profile.csv"
    )

    centroid_df = read_csv(
        CENTROID_DIR / f"{dataset}_centroid_analysis.csv"
    )

    ngram_overlap_df = read_csv(
        NGRAM_DIR / f"{dataset}_ngram_overlap.csv"
    )

    shared_ngrams_df = read_csv(
        NGRAM_DIR / f"{dataset}_shared_ngrams.csv"
    )

    create_sd_lineplots(dataset, linguistic_df, punctuation_df)
    create_mean_lineplots(dataset, linguistic_df, punctuation_df)
    create_violinplots(dataset, abstract_df)
    create_centroid_plots(dataset, centroid_df, abstract_df)
    create_ngram_heatmaps(dataset, ngram_overlap_df)
    create_ngram_drift_plots(dataset, ngram_overlap_df)
    create_ngram_late_period_convergence_plots(dataset, ngram_overlap_df)
    create_shared_ngram_tables(dataset, shared_ngrams_df)

    print("\nAll plots saved in:")
    print(PLOT_DIR)


# LOAD DATA
def read_csv(path):
    """
    Reads a CSV file and gives a clear error if it is missing.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    return pd.read_csv(path)


def create_plot_directories():
    """
    Creates plot output directories.
    """

    subdirs = [
        "sd_lineplots",
        "mean_lineplots",
        "violinplots",
        "centroid",
        "ngram_heatmaps",
        "ngram_drift",
        "ngram_late_convergence",
        "ngram_tables",
    ]

    for subdir in subdirs:
        (PLOT_DIR / subdir).mkdir(parents=True, exist_ok=True)


# LINEPLOTS




def create_sd_lineplots(dataset, linguistic_df, punctuation_df):
    """
    Creates SD lineplots for the feature categories used in the analysis:
    structural, lexical, grammatical, and punctuation-based features.
    """

    plot_multi_line(
        df=linguistic_df,
        columns=STRUCTURAL_SD_COLUMNS,
        title=f"{dataset}: Structural variation over time",
        ylabel="Standard deviation",
        output_path=(
            PLOT_DIR / "sd_lineplots" /
            f"{dataset}_structural_sd_lineplot.png"
        ),
    )

    plot_multi_line(
        df=linguistic_df,
        columns=LEXICAL_SD_COLUMNS,
        title=f"{dataset}: Lexical variation over time",
        ylabel="Standard deviation",
        output_path=(
            PLOT_DIR / "sd_lineplots" /
            f"{dataset}_lexical_sd_lineplot.png"
        ),
    )

    plot_multi_line(
        df=linguistic_df,
        columns=GRAMMATICAL_SD_COLUMNS,
        title=f"{dataset}: Grammatical variation over time",
        ylabel="Standard deviation",
        output_path=(
            PLOT_DIR / "sd_lineplots" /
            f"{dataset}_grammatical_sd_lineplot.png"
        ),
    )

    plot_multi_line(
        df=punctuation_df,
        columns=PUNCTUATION_SD_COLUMNS,
        title=f"{dataset}: Punctuation variation over time",
        ylabel="Standard deviation",
        output_path=(
            PLOT_DIR / "sd_lineplots" /
            f"{dataset}_punctuation_sd_lineplot.png"
        ),
    )


def create_mean_lineplots(dataset, linguistic_df, punctuation_df):
    """
    Creates mean lineplots for punctuation, POS, and lexical/structural metrics.
    """

    plot_multi_line(
        df=punctuation_df,
        columns=PUNCTUATION_MEAN_COLUMNS,
        title=f"{dataset}: Mean punctuation rates over time",
        ylabel="Mean value",
        output_path=(
            PLOT_DIR / "mean_lineplots" /
            f"{dataset}_punctuation_mean_lineplot.png"
        ),
    )

    plot_multi_line(
        df=linguistic_df,
        columns=POS_MEAN_COLUMNS,
        title=f"{dataset}: Mean POS ratios over time",
        ylabel="Mean value",
        output_path=(
            PLOT_DIR / "mean_lineplots" /
            f"{dataset}_pos_mean_lineplot.png"
        ),
    )

    plot_multi_line(
        df=linguistic_df,
        columns=LEXICAL_STRUCTURAL_MEAN_COLUMNS,
        title=f"{dataset}: Mean lexical and structural metrics over time",
        ylabel="Mean value",
        output_path=(
            PLOT_DIR / "mean_lineplots" /
            f"{dataset}_lexical_structural_mean_lineplot.png"
        ),
    )


def plot_multi_line(df, columns, title, ylabel, output_path):
    """
    Creates a multi-line plot for selected columns.
    """

    available_columns = [column for column in columns if column in df.columns]

    if not available_columns:
        print(f"Skipping plot, no columns found: {title}")
        return

    plt.figure(figsize=(14, 8))

    for column in available_columns:
        plt.plot(
            df["year"],
            df[column],
            marker="o",
            linewidth=2.5,
            label=clean_label(column),
        )

    plt.xlabel("Year")
    plt.ylabel(ylabel)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(available_columns),
        frameon=False,
        fontsize=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# VIOLINPLOTS
def create_violinplots(dataset, abstract_df):
    """
    Creates individual violinplots for all abstract-level metrics.
    """

    for column in VIOLIN_COLUMNS:
        if column not in abstract_df.columns:
            print(f"Skipping violinplot, missing column: {column}")
            continue

        output_path = (
            PLOT_DIR / "violinplots" /
            f"{dataset}_{column}_violinplot.png"
        )

        plot_violin_by_year(
            df=abstract_df,
            value_column=column,
            title=f"{dataset}: Distribution of {clean_label(column)} by year",
            ylabel=clean_label(column),
            output_path=output_path,
        )


def plot_violin_by_year(df, value_column, title, ylabel, output_path):
    """
    Creates a violinplot grouped by year.
    """

    years = sorted(df["year"].unique())

    data = [
        df[df["year"] == year][value_column].dropna().values
        for year in years
    ]

    plt.figure(figsize=(12, 6))
    plt.violinplot(data, showmeans=True, showmedians=True)

    plt.xticks(
        ticks=np.arange(1, len(years) + 1),
        labels=years,
        rotation=45,
    )

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# CENTROID PLOTS
def create_centroid_plots(dataset, centroid_df, abstract_df):
    """
    Creates centroid distance plots.
    """

    output_path = (
        PLOT_DIR / "centroid" /
        f"{dataset}_centroid_distance_mean_sd_lineplot.png"
    )

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        centroid_df["year"],
        centroid_df["mean_distance_to_centroid"],
        yerr=centroid_df["sd_distance_to_centroid"],
        marker="o",
        capsize=4,
    )

    # Mark significant adjacent-year increases
    for x in [2024.5, 2025.5]:
        plt.axvline(
            x=x,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    plt.text(
        2024.5,
        centroid_df["mean_distance_to_centroid"].max(),
        "*",
        ha="center",
        va="bottom",
        fontsize=22,
    )

    plt.text(
        2025.5,
        centroid_df["mean_distance_to_centroid"].max(),
        "*",
        ha="center",
        va="bottom",
        fontsize=22,
    )

    plt.title(f"{dataset}: Mean distance to yearly stylistic centroid")
    plt.xlabel("Year")
    plt.ylabel("Mean distance to centroid ± SD")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    if "distance_to_centroid" in abstract_df.columns:
        output_path = (
            PLOT_DIR / "centroid" /
            f"{dataset}_distance_to_centroid_violinplot.png"
        )

        plot_violin_by_year(
            df=abstract_df,
            value_column="distance_to_centroid",
            title=f"{dataset}: Distribution of distance to centroid by year",
            ylabel="Distance to centroid",
            output_path=output_path,
        )


# N-GRAM HEATMAPS
def create_ngram_heatmaps(dataset, overlap_df):
    """
    Creates Jaccard similarity heatmaps for each n-gram size and filter type.
    """

    ngram_labels = {
        1: "unigram",
        2: "bigram",
        3: "trigram",
    }

    for ngram_size in sorted(overlap_df["ngram_size"].unique()):
        for ngram_filter in sorted(overlap_df["ngram_filter"].unique()):

            subset = overlap_df[
                (overlap_df["ngram_size"] == ngram_size)
                & (overlap_df["ngram_filter"] == ngram_filter)
            ]

            if subset.empty:
                continue

            output_path = (
                PLOT_DIR / "ngram_heatmaps" /
                f"{dataset}_{ngram_labels[ngram_size]}_"
                f"{ngram_filter}_jaccard_heatmap.png"
            )

            plot_ngram_heatmap(
                subset=subset,
                title=(
                    f"{dataset}: {ngram_labels[ngram_size].title()} "
                    f"Jaccard similarity ({ngram_filter})"
                ),
                output_path=output_path,
            )


def plot_ngram_heatmap(subset, title, output_path):
    """
    Creates symmetric year-by-year heatmap from pairwise n-gram overlap table.
    """

    years = sorted(
        set(subset["year_a"].unique()).union(set(subset["year_b"].unique()))
    )

    matrix = pd.DataFrame(
        np.eye(len(years)),
        index=years,
        columns=years,
    )

    for _, row in subset.iterrows():
        year_a = row["year_a"]
        year_b = row["year_b"]
        value = row["jaccard_similarity"]

        matrix.loc[year_a, year_b] = value
        matrix.loc[year_b, year_a] = value

    plt.figure(figsize=(9, 8))
    plt.imshow(matrix.values, aspect="auto")

    plt.xticks(
        ticks=np.arange(len(years)),
        labels=years,
        rotation=45,
    )

    plt.yticks(
        ticks=np.arange(len(years)),
        labels=years,
    )

    plt.colorbar(label="Jaccard similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# N-GRAM TABLES
def create_shared_ngram_tables(dataset, shared_df):
    """
    Saves filtered shared n-gram tables by n-gram size and filter type.
    """

    ngram_labels = {
        1: "unigram",
        2: "bigram",
        3: "trigram",
    }

    for ngram_size in sorted(shared_df["ngram_size"].unique()):
        for ngram_filter in sorted(shared_df["ngram_filter"].unique()):

            subset = shared_df[
                (shared_df["ngram_size"] == ngram_size)
                & (shared_df["ngram_filter"] == ngram_filter)
            ].copy()

            if subset.empty:
                continue

            subset = subset.sort_values(
                by=[
                    "year_a",
                    "year_b",
                    "frequency_year_a",
                    "frequency_year_b",
                ],
                ascending=[True, True, False, False],
            )

            output_path = (
                PLOT_DIR / "ngram_tables" /
                f"{dataset}_shared_{ngram_labels[ngram_size]}_"
                f"{ngram_filter}.csv"
            )

            subset.to_csv(output_path, index=False)


def create_ngram_drift_plots(dataset, overlap_df):
    """
    Creates adjacent-year n-gram overlap plots.

    This shows how similar each year is to the following year:
    2016-2017, 2017-2018, etc.
    """

    for ngram_filter in sorted(overlap_df["ngram_filter"].unique()):

        subset = overlap_df[
            overlap_df["ngram_filter"] == ngram_filter
        ].copy()

        adjacent_df = subset[
            subset["year_b"] == subset["year_a"] + 1
        ].copy()

        if adjacent_df.empty:
            continue

        output_path = (
            PLOT_DIR / "ngram_drift" /
            f"{dataset}_{ngram_filter}_adjacent_year_overlap.png"
        )

        plot_ngram_line_comparison(
            df=adjacent_df,
            title=(
                f"{dataset}: Adjacent-year n-gram overlap "
                f"({ngram_filter})"
            ),
            ylabel="Jaccard similarity",
            output_path=output_path,
        )


def create_ngram_late_period_convergence_plots(dataset, overlap_df):
    """
    Creates late-period convergence plots.

    Each year is compared to the latest available year.
    If similarity increases toward the latest year, this suggests convergence
    toward a newer discourse pattern.
    """

    latest_year = max(
        overlap_df["year_a"].max(),
        overlap_df["year_b"].max(),
    )

    for ngram_filter in sorted(overlap_df["ngram_filter"].unique()):

        rows = []

        subset = overlap_df[
            overlap_df["ngram_filter"] == ngram_filter
        ].copy()

        for _, row in subset.iterrows():
            if row["year_a"] == latest_year:
                comparison_year = row["year_b"]
            elif row["year_b"] == latest_year:
                comparison_year = row["year_a"]
            else:
                continue

            rows.append(
                {
                    "comparison_year": comparison_year,
                    "ngram_size": row["ngram_size"],
                    "jaccard_similarity": row["jaccard_similarity"],
                }
            )

        convergence_df = pd.DataFrame(rows)

        if convergence_df.empty:
            continue

        convergence_df = convergence_df.sort_values(
            by=["comparison_year", "ngram_size"]
        )

        output_path = (
            PLOT_DIR / "ngram_late_convergence" /
            f"{dataset}_{ngram_filter}_late_period_convergence.png"
        )

        plot_late_period_convergence(
            df=convergence_df,
            latest_year=latest_year,
            title=(
                f"{dataset}: N-gram similarity to {latest_year} "
                f"({ngram_filter})"
            ),
            ylabel=f"Jaccard similarity to {latest_year}",
            output_path=output_path,
        )


def plot_ngram_line_comparison(df, title, ylabel, output_path):
    """
    Plots unigram, bigram, and trigram Jaccard similarity over adjacent years.
    """

    ngram_labels = {
        1: "unigram",
        2: "bigram",
        3: "trigram",
    }

    plt.figure(figsize=(11, 6))

    for ngram_size in sorted(df["ngram_size"].unique()):
        size_df = df[df["ngram_size"] == ngram_size].copy()

        size_df["year_pair"] = (
            size_df["year_a"].astype(str)
            + "-"
            + size_df["year_b"].astype(str)
        )

        plt.plot(
            size_df["year_pair"],
            size_df["jaccard_similarity"],
            marker="o",
            label=ngram_labels.get(ngram_size, str(ngram_size)),
        )

    plt.title(title)
    plt.xlabel("Adjacent year pair")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title="N-gram size")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_late_period_convergence(
    df,
    latest_year,
    title,
    ylabel,
    output_path,
):
    """
    Plots similarity between each earlier year and the latest year.
    """

    ngram_labels = {
        1: "unigram",
        2: "bigram",
        3: "trigram",
    }

    plt.figure(figsize=(11, 6))

    for ngram_size in sorted(df["ngram_size"].unique()):
        size_df = df[df["ngram_size"] == ngram_size].copy()

        plt.plot(
            size_df["comparison_year"],
            size_df["jaccard_similarity"],
            marker="o",
            label=ngram_labels.get(ngram_size, str(ngram_size)),
        )

    plt.title(title)
    plt.xlabel(f"Year compared with {latest_year}")
    plt.ylabel(ylabel)
    plt.legend(title="N-gram size")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# HELPERS
def clean_label(label):
    """
    Converts variable names into readable plot labels.
    """

    return label.replace("_", " ")


# PARSER
def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Create plots from stored analysis outputs."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="quant-ph",
        choices=["cs.CL", "math-ph", "quant-ph"],
        help="Dataset to plot. Default: quant-ph",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()