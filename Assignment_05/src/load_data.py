"""
load_data.py

Loads, filters, and samples arXiv metadata for Assignment 05.

The script:
1. Downloads the arXiv metadata file from Kaggle
2. Reads the raw JSON-lines metadata file
3. Extracts relevant metadata: id, title, abstract, categories, and year
4. Filters papers from selected categories and years
5. Computes abstract word counts within each category-year group
6. Removes abstracts below the first quartile of abstract length within each group
7. Automatically determines the maximum balanced sample size per category
8. Randomly samples the same number of papers per year within each category
9. Saves one CSV file per category in the in/ folder
"""

# IMPORTS
import json
import random
from pathlib import Path

import kagglehub
import pandas as pd
from tqdm import tqdm


# PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent

IN_DIR = ROOT_DIR / "in"

DATASET_HANDLE = "Cornell-University/arxiv"
DATASET_FILE = "arxiv-metadata-oai-snapshot.json"

START_YEAR = 2016
END_YEAR = 2026

RANDOM_SEED = 42

CATEGORY_CONFIG = {
    "cs.CL": {
        "output_file": IN_DIR / "cs_cl_2016_2026.csv",
    },
    "math-ph": {
        "output_file": IN_DIR / "math_ph_2016_2026.csv",
    },
    "quant-ph": {
        "output_file": IN_DIR / "quant_ph_2016_2026.csv",
    },
}


# FUNCTION: MAIN
def main():
    random.seed(RANDOM_SEED)

    dataset_path = download_dataset()
    raw_file = dataset_path / DATASET_FILE

    category_records = collect_category_records(
        filepath=raw_file,
        category_config=CATEGORY_CONFIG,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    sampled_datasets = sample_category_records(
        category_records=category_records,
    )

    save_datasets(sampled_datasets, CATEGORY_CONFIG)

    print_summary(sampled_datasets)


# FUNCTION: SUPPORT
def download_dataset():
    """
    Downloads the latest version of the Kaggle arXiv dataset.
    Returns the local dataset folder path.
    """

    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)

    return Path(dataset_path)


def collect_category_records(filepath, category_config, start_year, end_year):
    """
    Reads the arXiv JSON-lines file and collects papers matching selected categories.

    Papers can belong to multiple arXiv categories. If a paper belongs to more than
    one selected category, it is included in each matching category dataset.
    """

    selected_categories = set(category_config.keys())

    category_records = {
        category: {year: [] for year in range(start_year, end_year + 1)}
        for category in selected_categories
    }

    with open(filepath, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Collecting selected arXiv records"):
            paper = json.loads(line)

            year = extract_year(paper)

            if year is None:
                continue

            if not (start_year <= year <= end_year):
                continue

            paper_categories = extract_categories(paper)
            matching_categories = selected_categories.intersection(paper_categories)

            if not matching_categories:
                continue

            abstract = paper.get("abstract", "")

            if not abstract.strip():
                continue

            record = {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "abstract": abstract,
                "categories": paper.get("categories"),
                "year": year,
            }

            for category in matching_categories:
                category_records[category][year].append(record)

    return category_records


def sample_category_records(category_records):
    """
    Randomly samples a balanced number of papers per year for each category.

    For each category-year group, abstracts below the first quartile of abstract
    length are removed. The final sample size for each category is determined by
    the year with the fewest remaining abstracts after Q1 filtering.
    """

    sampled_datasets = {}

    for category, yearly_records in category_records.items():
        print("\n" + "=" * 80)
        print(f"PROCESSING CATEGORY: {category}")
        print("=" * 80)

        filtered_yearly_dfs = {}
        filtering_info = []

        for year, records in sorted(yearly_records.items()):
            df_year = pd.DataFrame(records)

            if df_year.empty:
                raise ValueError(
                    f"No papers found for {category} in {year}."
                )

            df_year["abstract_word_count"] = count_abstract_words(
                df_year["abstract"]
            )

            n_before_filter = len(df_year)
            q1_threshold = df_year["abstract_word_count"].quantile(0.25)

            df_year = df_year[
                df_year["abstract_word_count"] >= q1_threshold
            ].copy()

            n_after_filter = len(df_year)

            df_year["q1_length_threshold"] = q1_threshold
            df_year["n_available_after_q1_filter"] = n_after_filter

            filtered_yearly_dfs[year] = df_year

            filtering_info.append(
                {
                    "year": year,
                    "n_before_q1_filter": n_before_filter,
                    "q1_length_threshold": q1_threshold,
                    "n_after_q1_filter": n_after_filter,
                }
            )

        filtering_df = pd.DataFrame(filtering_info).sort_values("year")

        sample_per_year = int(filtering_df["n_after_q1_filter"].min())

        limiting_year = filtering_df.loc[
            filtering_df["n_after_q1_filter"].idxmin(),
            "year",
        ]

        print("\nQ1 filtering overview:")
        print(filtering_df.to_string(index=False))

        print("\nBalanced sampling decision:")
        print(f"Limiting year: {limiting_year}")
        print(f"Sample per year for {category}: {sample_per_year}")

        sampled_records = []

        for year, df_year in sorted(filtered_yearly_dfs.items()):
            sampled_df = df_year.sample(
                n=sample_per_year,
                random_state=RANDOM_SEED + year,
            )

            sampled_df["sample_per_year"] = sample_per_year
            sampled_df["limiting_year"] = limiting_year

            sampled_records.append(sampled_df)

            print(
                f"Extracted {len(sampled_df)} papers from {category} in {year} "
                f"(available after Q1 filter: {len(df_year)})"
            )

        sampled_datasets[category] = pd.concat(
            sampled_records,
            ignore_index=True,
        )

    return sampled_datasets


def count_abstract_words(abstract_series):
    """
    Counts words in a pandas Series of abstracts.
    """

    return (
        abstract_series
        .astype(str)
        .str.split()
        .str.len()
    )


def save_datasets(sampled_datasets, category_config):
    """
    Saves one sampled dataframe per category.
    """

    IN_DIR.mkdir(parents=True, exist_ok=True)

    for category, df in sampled_datasets.items():
        output_file = category_config[category]["output_file"]

        df.to_csv(output_file, index=False)

        print(f"\nSaved {category} dataset to:")
        print(output_file)


def print_summary(sampled_datasets):
    """
    Prints a short summary of the saved datasets.
    """

    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)

    for category, df in sampled_datasets.items():
        print("\n" + "-" * 80)
        print(f"Category: {category}")
        print(f"Total papers: {len(df)}")
        print(f"Sample per year: {df['sample_per_year'].iloc[0]}")
        print(f"Limiting year: {df['limiting_year'].iloc[0]}")

        print("\nPapers per year:")
        print(df["year"].value_counts().sort_index().to_string())

        print("\nAbstract word count:")
        print(df["abstract_word_count"].describe().to_string())

        print("\nQ1 thresholds by year:")
        print(
            df.groupby("year")["q1_length_threshold"]
            .first()
            .sort_index()
            .to_string()
        )


def extract_year(paper):
    """
    Extracts year from the first arXiv version date.
    """

    try:
        created = paper["versions"][0]["created"]

        # Example:
        # "Mon, 31 Mar 2008 14:00:00 GMT"
        year = created.split()[3]

        return int(year)

    except (KeyError, IndexError, TypeError, ValueError):
        return None


def extract_categories(paper):
    """
    Extracts arXiv categories as a list.
    """

    categories = paper.get("categories", "")

    if not categories:
        return []

    return categories.split()


# FUNCTION: PARSER


# CALL FOR MAIN
if __name__ == "__main__":
    main()