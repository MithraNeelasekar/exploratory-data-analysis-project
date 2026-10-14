"""
Automated Test Suite for Exploratory Data Analysis (EDA) Project.

Tests all analytical functions, cleaning routines, outlier methods,
visual generation, and the 13-heading report generator.
"""

import os
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test environments

from data_cleaning import (
    get_missing_summary,
    get_duplicate_summary,
    remove_duplicates,
    impute_numerical_column,
    impute_categorical_column,
    detect_outliers_iqr,
    handle_outliers_iqr,
    convert_column_type,
    get_cleaning_diff
)
from eda_analysis import (
    get_dataset_overview,
    get_column_types,
    get_numerical_summary,
    get_categorical_summary,
    get_frequency_distribution,
    compute_correlation_matrix,
    get_strongest_correlations,
    analyze_target_relationship,
    generate_key_findings
)
from visualizations import (
    plot_missing_values,
    plot_numerical_distribution,
    plot_boxplot,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_scatter,
    plot_bivariate_categorical,
    plot_pairplot
)
from report_generator import generate_eda_markdown_report


class TestEDACleaning(unittest.TestCase):
    """Unit tests for data cleaning functions."""

    def setUp(self):
        self.df = pd.DataFrame({
            "age": [20.0, 30.0, np.nan, 50.0, 100.0, 60.0],  # 1 missing, 1 outlier (100)
            "gender": ["male", "female", "female", np.nan, "female", "male"],
            "income": [1000, 2000, 3000, 4000, 50000, 6000],
            "city": ["NY", "LA", "NY", "SF", "NY", "BOS"]
        })
        # Add a known duplicate row
        self.df_with_dups = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)

    def test_missing_summary(self):
        summary = get_missing_summary(self.df)
        self.assertEqual(len(summary), 4)
        age_missing = summary[summary["Column"] == "age"]["Missing Count"].values[0]
        self.assertEqual(age_missing, 1)

    def test_duplicate_summary_and_removal(self):
        dup_info = get_duplicate_summary(self.df_with_dups)
        self.assertTrue(dup_info["has_duplicates"])
        self.assertEqual(dup_info["duplicate_count"], 1)

        deduped, dropped = remove_duplicates(self.df_with_dups)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(deduped), len(self.df))

    def test_impute_numerical_column(self):
        # Median imputation
        df_med, audit = impute_numerical_column(self.df, "age", strategy="median")
        self.assertEqual(df_med["age"].isna().sum(), 0)
        self.assertEqual(audit["imputed_count"], 1)

        # Mean imputation
        df_mean, _ = impute_numerical_column(self.df, "age", strategy="mean")
        self.assertEqual(df_mean["age"].isna().sum(), 0)

        # Constant imputation
        df_const, _ = impute_numerical_column(self.df, "age", strategy="constant", custom_value=99.0)
        self.assertEqual(df_const.loc[2, "age"], 99.0)

        # Drop rows
        df_drop, _ = impute_numerical_column(self.df, "age", strategy="drop")
        self.assertEqual(len(df_drop), len(self.df) - 1)

    def test_impute_categorical_column(self):
        # Mode imputation
        df_mode, _ = impute_categorical_column(self.df, "gender", strategy="mode")
        self.assertEqual(df_mode["gender"].isna().sum(), 0)
        self.assertEqual(df_mode.loc[3, "gender"], "female")

        # Constant imputation
        df_const, _ = impute_categorical_column(self.df, "gender", strategy="constant", custom_value="Unknown")
        self.assertEqual(df_const.loc[3, "gender"], "Unknown")

    def test_detect_outliers_iqr(self):
        # Data: 10, 11, 12, 13, 14, 100 -> 100 is an outlier
        test_df = pd.DataFrame({"val": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0]})
        outlier_res = detect_outliers_iqr(test_df, "val", multiplier=1.5)
        self.assertEqual(outlier_res["outlier_count"], 1)
        self.assertGreater(outlier_res["upper_bound"], 14.0)

    def test_handle_outliers_iqr_clip(self):
        test_df = pd.DataFrame({"val": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0]})
        clipped_df, audit = handle_outliers_iqr(test_df, "val", action="clip", multiplier=1.5)
        self.assertEqual(len(clipped_df), len(test_df))
        self.assertLess(clipped_df["val"].max(), 100.0)

    def test_handle_outliers_iqr_drop(self):
        test_df = pd.DataFrame({"val": [10.0, 11.0, 12.0, 13.0, 14.0, 100.0]})
        dropped_df, audit = handle_outliers_iqr(test_df, "val", action="drop", multiplier=1.5)
        self.assertEqual(len(dropped_df), len(test_df) - 1)

    def test_convert_column_type(self):
        c_df, success, _ = convert_column_type(self.df, "income", "string")
        self.assertTrue(success)
        self.assertTrue(pd.api.types.is_string_dtype(c_df["income"]))

    def test_cleaning_diff(self):
        cleaned, _ = impute_numerical_column(self.df, "age", strategy="median")
        diff = get_cleaning_diff(self.df, cleaned)
        self.assertEqual(diff["missing_before"], 2)
        self.assertEqual(diff["missing_after"], 1)


class TestEDAAnalysis(unittest.TestCase):
    """Unit tests for statistical analysis computations."""

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "num2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "num_inv": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "cat1": ["A", "A", "B", "B", "A", "C", "C", "A", "B", "A"],
            "target_bin": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
        })

    def test_overview(self):
        overview = get_dataset_overview(self.df)
        self.assertEqual(overview["total_rows"], 10)
        self.assertEqual(overview["total_columns"], 5)
        self.assertEqual(overview["total_missing"], 0)

    def test_numerical_summary(self):
        summary = get_numerical_summary(self.df)
        self.assertEqual(len(summary), 4)  # num1, num2, num_inv, target_bin
        num1_stats = summary[summary["Column"] == "num1"].iloc[0]
        self.assertEqual(num1_stats["Mean"], 5.5)
        self.assertEqual(num1_stats["Min"], 1.0)
        self.assertEqual(num1_stats["Max"], 10.0)

    def test_categorical_summary(self):
        cat_summary = get_categorical_summary(self.df)
        self.assertEqual(len(cat_summary), 1)
        row = cat_summary.iloc[0]
        self.assertEqual(row["Column"], "cat1")
        self.assertEqual(row["Mode (Most Frequent)"], "A")
        self.assertEqual(row["Mode Frequency"], 5)

    def test_frequency_distribution(self):
        dist = get_frequency_distribution(self.df, "cat1")
        self.assertEqual(dist["Count"].sum(), 10)
        self.assertAlmostEqual(dist["Percentage (%)"].sum(), 100.0, places=1)

    def test_correlation_matrix_and_strongest(self):
        corr = compute_correlation_matrix(self.df, method="pearson")
        self.assertAlmostEqual(corr.loc["num1", "num2"], 1.0, places=2)
        self.assertAlmostEqual(corr.loc["num1", "num_inv"], -1.0, places=2)

        strong = get_strongest_correlations(corr, top_n=2)
        pos = strong["positive"]
        neg = strong["negative"]
        self.assertFalse(pos.empty)
        self.assertFalse(neg.empty)
        self.assertEqual(pos.iloc[0]["Strength"], "Strong")
        self.assertEqual(neg.iloc[0]["Strength"], "Strong")

    def test_target_relationship(self):
        # Num vs Num
        res_num = analyze_target_relationship(self.df, "num2", "num1")
        self.assertEqual(res_num["type"], "num_vs_num")
        self.assertAlmostEqual(res_num["pearson_r"], 1.0, places=2)

        # Cat target vs Num feature
        res_cat_num = analyze_target_relationship(self.df, "target_bin", "num1")
        self.assertEqual(res_cat_num["type"], "cat_target_vs_num_feature")
        self.assertIn("group_table", res_cat_num)

        # Cat vs Cat
        res_cat = analyze_target_relationship(self.df, "target_bin", "cat1")
        self.assertEqual(res_cat["type"], "cat_vs_cat")
        self.assertIn("contingency_table", res_cat)

    def test_generate_key_findings(self):
        findings = generate_key_findings(self.df, target_col="target_bin")
        self.assertIsInstance(findings, list)
        self.assertGreater(len(findings), 0)


class TestVisualizations(unittest.TestCase):
    """Unit tests for figure generation functions."""

    def setUp(self):
        self.df = pd.DataFrame({
            "age": [22, 38, 26, 35, 54, 2, 27, 14, 4, 58],
            "fare": [7.25, 71.28, 7.92, 53.10, 8.05, 8.45, 51.86, 21.07, 11.13, 30.07],
            "sex": ["male", "female", "female", "female", "male", "male", "male", "female", "female", "female"],
            "survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        })

    def test_plot_missing_values(self):
        fig = plot_missing_values(self.df)
        self.assertIsNotNone(fig)

    def test_plot_numerical_distribution(self):
        fig = plot_numerical_distribution(self.df, "age", bins=10)
        self.assertIsNotNone(fig)

    def test_plot_boxplot(self):
        fig = plot_boxplot(self.df, "fare", group_by="sex")
        self.assertIsNotNone(fig)

    def test_plot_categorical_distribution(self):
        fig = plot_categorical_distribution(self.df, "sex")
        self.assertIsNotNone(fig)

    def test_plot_correlation_heatmap(self):
        corr = compute_correlation_matrix(self.df)
        fig = plot_correlation_heatmap(corr)
        self.assertIsNotNone(fig)

    def test_plot_scatter(self):
        fig = plot_scatter(self.df, "age", "fare", color_col="sex")
        self.assertIsNotNone(fig)

    def test_plot_bivariate_categorical(self):
        fig = plot_bivariate_categorical(self.df, "sex", "survived")
        self.assertIsNotNone(fig)

    def test_plot_pairplot(self):
        fig = plot_pairplot(self.df, ["age", "fare"], hue_col="sex")
        self.assertIsNotNone(fig)


class TestReportGenerator(unittest.TestCase):
    """Unit tests for the 13-heading academic EDA report generator."""

    def setUp(self):
        csv_path = os.path.join(os.path.dirname(__file__), "data", "titanic.csv")
        self.df = pd.read_csv(csv_path)

    def test_report_headings_present(self):
        report = generate_eda_markdown_report(
            df_raw=self.df,
            df_cleaned=self.df,
            cleaning_history=["Imputed Age with median"],
            dataset_name="Titanic Dataset",
            target_col="Survived"
        )

        expected_headings = [
            "# 1. Project Title",
            "## 2. Objective",
            "## 3. Dataset Description",
            "## 4. Data Overview",
            "## 5. Data Quality and Missing Values",
            "## 6. Data Cleaning Performed",
            "## 7. Statistical Summary",
            "## 8. Important Visualizations",
            "## 9. Correlation Analysis",
            "## 10. Key Findings",
            "## 11. Limitations",
            "## 12. Conclusion",
            "## 13. Future Improvements"
        ]

        for heading in expected_headings:
            self.assertIn(heading, report, f"Missing required heading: {heading}")

        # Check that actual numbers are inserted
        self.assertIn("891", report)
        self.assertIn("Titanic", report)


class TestEDAEdgeCases(unittest.TestCase):
    """Unit tests for extreme and boundary edge cases."""

    def test_all_numeric_dataset(self):
        df_num = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        overview = get_dataset_overview(df_num)
        self.assertEqual(overview["cat_count"], 0)
        self.assertEqual(overview["num_count"], 2)
        corr = compute_correlation_matrix(df_num)
        self.assertFalse(corr.empty)

    def test_all_categorical_dataset(self):
        df_cat = pd.DataFrame({"color": ["red", "blue", "red"], "size": ["S", "M", "L"]})
        overview = get_dataset_overview(df_cat)
        self.assertEqual(overview["num_count"], 0)
        self.assertEqual(overview["cat_count"], 2)
        corr = compute_correlation_matrix(df_cat)
        self.assertTrue(corr.empty)
        num_summary = get_numerical_summary(df_cat)
        self.assertTrue(num_summary.empty)

    def test_empty_dataframe(self):
        df_empty = pd.DataFrame()
        overview = get_dataset_overview(df_empty)
        self.assertEqual(overview["total_rows"], 0)
        self.assertEqual(overview["total_columns"], 0)

    def test_no_outliers(self):
        df_flat = pd.DataFrame({"val": [10.0, 10.0, 10.0, 10.0, 10.0]})
        res = detect_outliers_iqr(df_flat, "val")
        self.assertEqual(res["outlier_count"], 0)


if __name__ == "__main__":
    unittest.main()
