"""
centroid_analysis.py

Creates a multidimensional stylistic dispersion analysis.

The analysis represents each abstract as a stylistic feature vector.
For each year, it calculates the distance from each abstract to that year's
stylistic centroid.

Lower average distance to centroid indicates lower internal stylistic variation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests


CENTROID_DIR_NAME = "centroid"

STYLE_FEATURES = [
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
]


def create_centroid_analysis(df, dataset_name, root_dir):
    """
    Creates and saves yearly distance-to-centroid analysis.
    """

    df = df.copy()

    validate_style_features(df)

    df = add_scaled_style_features(df)

    df = add_distance_to_yearly_centroid(df)

    centroid_df = aggregate_centroid_distances(df)

    anova_df = create_centroid_anova(df, dataset_name, root_dir)

    adjacent_df = create_adjacent_year_tests(df, dataset_name, root_dir)

    save_centroid_analysis(centroid_df, dataset_name, root_dir)

    print_centroid_preview(centroid_df)

    return df, centroid_df


def validate_style_features(df):
    """
    Checks that all required style features exist in the dataframe.
    """

    missing_features = [
        feature for feature in STYLE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required style features for centroid analysis: "
            + ", ".join(missing_features)
        )


def add_scaled_style_features(df):
    """
    Standardizes stylistic features using z-score scaling.

    Scaling is necessary because the features are measured on different scales.
    """

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df[STYLE_FEATURES])

    for index, feature in enumerate(STYLE_FEATURES):
        df[f"scaled_{feature}"] = scaled_values[:, index]

    return df


def add_distance_to_yearly_centroid(df):
    """
    Calculates each abstract's distance to its yearly stylistic centroid.
    """

    scaled_features = [
        f"scaled_{feature}"
        for feature in STYLE_FEATURES
    ]

    df["distance_to_centroid"] = np.nan

    for year, group in df.groupby("year"):
        centroid = group[scaled_features].mean().values

        distances = np.linalg.norm(
            group[scaled_features].values - centroid,
            axis=1,
        )

        df.loc[group.index, "distance_to_centroid"] = distances

    return df


def aggregate_centroid_distances(df):
    """
    Aggregates distance-to-centroid values by year.
    """

    centroid_df = (
        df.groupby("year")
        .agg(
            texts=("distance_to_centroid", "count"),
            mean_distance_to_centroid=(
                "distance_to_centroid",
                "mean",
            ),
            sd_distance_to_centroid=(
                "distance_to_centroid",
                "std",
            ),
            median_distance_to_centroid=(
                "distance_to_centroid",
                "median",
            ),
        )
        .reset_index()
    )

    centroid_df = centroid_df.round(4)

    return centroid_df


def save_centroid_analysis(centroid_df, dataset_name, root_dir):
    """
    Saves yearly centroid analysis as CSV.
    """

    output_dir = Path(root_dir) / "out" / CENTROID_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_centroid_analysis.csv"

    centroid_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("CENTROID ANALYSIS SAVED")
    print("=" * 80)
    print(f"\nSaved to:\n{output_path}")


def print_centroid_preview(centroid_df):
    """
    Prints preview of centroid analysis.
    """

    print("\nPreview:\n")
    print(centroid_df.to_string(index=False))

def create_centroid_anova(df, dataset_name, root_dir):
    """
    Performs one-way ANOVA to test whether centroid distances differ by year.
    """

    groups = [
        group["distance_to_centroid"].dropna().values
        for _, group in df.groupby("year")
    ]

    f_statistic, p_value = f_oneway(*groups)

    n = df["distance_to_centroid"].dropna().shape[0]
    k = df["year"].nunique()

    df_between = k - 1
    df_within = n - k

    anova_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "test": "one-way ANOVA",
                "dependent_variable": "distance_to_centroid",
                "grouping_variable": "year",
                "df_between": df_between,
                "df_within": df_within,
                "F": round(f_statistic, 4),
                "p": round(p_value, 6),
            }
        ]
    )

    save_centroid_anova(anova_df, dataset_name, root_dir)

    print_centroid_anova(anova_df)

    return anova_df


def save_centroid_anova(anova_df, dataset_name, root_dir):
    """
    Saves centroid ANOVA results as CSV.
    """

    output_dir = Path(root_dir) / "out" / CENTROID_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_centroid_anova.csv"

    anova_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("CENTROID ANOVA SAVED")
    print("=" * 80)
    print(f"\nSaved to:\n{output_path}")


def print_centroid_anova(anova_df):
    """
    Prints centroid ANOVA results.
    """

    print("\nANOVA result:\n")
    print(anova_df.to_string(index=False))

def create_adjacent_year_tests(df, dataset_name, root_dir):
    """
    Performs independent t-tests between adjacent publication years.

    P-values are adjusted using the Holm-Bonferroni correction.
    """

    results = []

    years = sorted(df["year"].dropna().unique())

    for year_1, year_2 in zip(years[:-1], years[1:]):
        group_1 = df.loc[
            df["year"] == year_1,
            "distance_to_centroid",
        ].dropna()

        group_2 = df.loc[
            df["year"] == year_2,
            "distance_to_centroid",
        ].dropna()

        t_statistic, p_value = ttest_ind(
            group_1,
            group_2,
            equal_var=False,
        )

        results.append(
            {
                "dataset": dataset_name,
                "comparison": f"{year_1}-{year_2}",
                "year_1": year_1,
                "year_2": year_2,
                "mean_year_1": round(group_1.mean(), 4),
                "mean_year_2": round(group_2.mean(), 4),
                "mean_difference": round(group_2.mean() - group_1.mean(), 4),
                "t": round(t_statistic, 4),
                "p_raw": p_value,
            }
        )

    adjacent_df = pd.DataFrame(results)

    rejected, p_adjusted, _, _ = multipletests(
        adjacent_df["p_raw"],
        alpha=0.05,
        method="holm",
    )

    adjacent_df["p_adjusted_holm"] = p_adjusted
    adjacent_df["significant_holm"] = rejected

    adjacent_df["p_raw"] = adjacent_df["p_raw"].round(6)
    adjacent_df["p_adjusted_holm"] = adjacent_df[
        "p_adjusted_holm"
    ].round(6)

    save_adjacent_year_tests(adjacent_df, dataset_name, root_dir)

    print_adjacent_year_tests(adjacent_df)

    return adjacent_df

def save_adjacent_year_tests(adjacent_df, dataset_name, root_dir):
    """
    Saves adjacent-year t-test results as CSV.
    """

    output_dir = Path(root_dir) / "out" / CENTROID_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_adjacent_year_tests.csv"

    adjacent_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("ADJACENT-YEAR TESTS SAVED")
    print("=" * 80)
    print(f"\nSaved to:\n{output_path}")


def print_adjacent_year_tests(adjacent_df):
    """
    Prints adjacent-year t-test results.
    """

    print("\nAdjacent-year t-test results:\n")
    print(adjacent_df.to_string(index=False))