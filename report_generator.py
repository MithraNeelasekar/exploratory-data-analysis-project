"""
Structured EDA Report Generator.

Dynamically compiles a rigorous, academic-grade Exploratory Data Analysis report
strictly conforming to the 13 required headings using real computed metrics.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from eda_analysis import (
    get_dataset_overview,
    get_numerical_summary,
    get_categorical_summary,
    compute_correlation_matrix,
    get_strongest_correlations,
    generate_key_findings
)
from data_cleaning import (
    get_missing_summary,
    get_duplicate_summary
)


def generate_eda_markdown_report(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    cleaning_history: List[str],
    dataset_name: str = "Titanic Dataset",
    target_col: Optional[str] = None
) -> str:
    """
    Constructs a complete Markdown report following the 13 required sections.

    Args:
        df_raw: Original raw DataFrame.
        df_cleaned: Cleaned DataFrame currently in use.
        cleaning_history: List of cleaning actions executed during the session.
        dataset_name: Name or file source of the dataset.
        target_col: Selected target column for correlation/relationship inquiry.

    Returns:
        Full markdown string ready for rendering and downloading.
    """
    overview_raw = get_dataset_overview(df_raw)
    overview_clean = get_dataset_overview(df_cleaned)
    missing_summary = get_missing_summary(df_cleaned)
    dup_info = get_duplicate_summary(df_cleaned)
    num_summary = get_numerical_summary(df_cleaned)
    cat_summary = get_categorical_summary(df_cleaned)
    corr_matrix = compute_correlation_matrix(df_cleaned, method="pearson")
    strong_corrs = get_strongest_correlations(corr_matrix, top_n=5)
    key_findings = generate_key_findings(df_cleaned, target_col=target_col)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md: List[str] = []

    # 1. Project Title
    md.append("# 1. Project Title")
    md.append("**Exploratory Data Analysis (EDA) Project**  ")
    md.append(f"*Automated Analytical Report | Dataset: {dataset_name} | Generated: {timestamp_str}*\n")
    md.append("> **Academic Notice:** All statistics, counts, and correlation coefficients presented in this report "
              "are dynamically computed from the active dataset. Automated interpretations represent preliminary findings "
              "and should be verified through domain knowledge.\n")

    # 2. Objective
    md.append("## 2. Objective")
    md.append(
        "The objective of this exploratory data analysis is to rigorously examine the dataset to uncover underlying "
        "distributions, assess data hygiene, identify meaningful bivariate and multivariate correlations, "
        "and establish evidence-based hypotheses regarding key influencing factors.\n"
    )

    # 3. Dataset Description
    md.append("## 3. Dataset Description")
    if "titanic" in dataset_name.lower():
        md.append(
            f"The **{dataset_name}** captures passenger demographics, travel accommodations, and survival outcomes "
            "from the historic 1912 maritime voyage of the RMS Titanic. It provides a standard benchmark for multi-type "
            "data exploration combining continuous variables (e.g., Age, Fare), discrete counts (e.g., SibSp, Parch), "
            "and categorical indicators (e.g., Sex, Class, EmbarkTown, Survived).\n"
        )
    else:
        md.append(
            f"The analyzed dataset is **{dataset_name}**, uploaded by the user for custom exploratory analysis. "
            f"It encompasses {overview_clean['num_count']} numerical variables and {overview_clean['cat_count']} "
            "categorical/text fields.\n"
        )

    # 4. Data Overview
    md.append("## 4. Data Overview")
    md.append(f"- **Total Rows (Observations):** {overview_clean['total_rows']:,}")
    md.append(f"- **Total Columns (Features):** {overview_clean['total_columns']}")
    md.append(f"- **Numerical Columns:** {len(overview_clean['numerical_columns'])} ({', '.join(overview_clean['numerical_columns']) if overview_clean['numerical_columns'] else 'None'})")
    md.append(f"- **Categorical Columns:** {len(overview_clean['categorical_columns'])} ({', '.join(overview_clean['categorical_columns']) if overview_clean['categorical_columns'] else 'None'})")
    md.append(f"- **Estimated Memory Footprint:** {overview_clean['memory_usage_mb']} MB\n")

    # 5. Data Quality and Missing Values
    md.append("## 5. Data Quality and Missing Values")
    md.append(f"- **Total Missing Cells:** {overview_clean['total_missing']:,} ({overview_clean['missing_percentage']}% of all cells)")
    md.append(f"- **Duplicate Rows:** {dup_info['duplicate_count']} ({dup_info['duplicate_pct']}% of records)\n")

    cols_with_missing = missing_summary[missing_summary["Missing Count"] > 0]
    if not cols_with_missing.empty:
        md.append("### Summary of Columns with Missing Data")
        md.append("| Column | Data Type | Missing Count | Missing (%) | Non-Null Count |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for _, row in cols_with_missing.iterrows():
            md.append(f"| **{row['Column']}** | `{row['Data Type']}` | {int(row['Missing Count']):,} | {row['Missing (%)']}% | {int(row['Non-Null Count']):,} |")
        md.append("")
    else:
        md.append("✅ **Clean Record Audit:** No missing values exist in the active dataset.\n")

    # 6. Data Cleaning Performed
    md.append("## 6. Data Cleaning Performed")
    if cleaning_history:
        md.append("The following data cleaning interventions were executed on the working copy:")
        for idx, action in enumerate(cleaning_history, 1):
            md.append(f"{idx}. {action}")
        md.append("")
        md.append(f"**Net Change:** Rows went from {overview_raw['total_rows']:,} to {overview_clean['total_rows']:,}; "
                  f"missing cells went from {overview_raw['total_missing']:,} to {overview_clean['total_missing']:,}.\n")
    else:
        md.append("No cleaning transformations were applied. Analysis reflects the raw imported data.\n")

    # 7. Statistical Summary
    md.append("## 7. Statistical Summary")
    if not num_summary.empty:
        md.append("### Numerical Features Descriptive Statistics")
        md.append("| Feature | Count | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max | Skewness |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in num_summary.iterrows():
            md.append(f"| **{row['Column']}** | {int(row['Count']):,} | {row['Mean']} | {row['Std Dev']} | {row['Min']} | "
                      f"{row['25% (Q1)']} | {row['50% (Median)']} | {row['75% (Q3)']} | {row['Max']} | {row['Skewness']} |")
        md.append("")

    if not cat_summary.empty:
        md.append("### Categorical Features Summary")
        md.append("| Feature | Non-Null | Unique | Mode (Most Common) | Mode Count | Mode (%) |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for _, row in cat_summary.iterrows():
            md.append(f"| **{row['Column']}** | {int(row['Non-Null Count']):,} | {int(row['Unique Values'])} | "
                      f"`{row['Mode (Most Frequent)']}` | {int(row['Mode Frequency']):,} | {row['Mode (%)']}% |")
        md.append("")

    # 8. Important Visualizations
    md.append("## 8. Important Visualizations")
    md.append(
        "During visual exploration in the dashboard, the following core visualization paradigms were deployed:\n"
        "- **Histograms & Density Estimates:** Evaluated central tendency, unimodal/bimodal shapes, and long-tail behavior.\n"
        "- **Box & Whisker Plots:** Pinpointed values falling beyond 1.5x IQR boundaries, revealing statistical anomalies.\n"
        "- **Categorical Frequency Plots:** Displayed class dominance and representation disparities across subgroups.\n"
        "- **Scatter Plots & OLS Trendlines:** Inspected linear co-variation and dispersion between continuous attributes.\n"
        "- **Annotated Correlation Heatmaps:** Mapped pairwise Pearson and Spearman correlation magnitudes across the feature space.\n"
    )

    # 9. Correlation Analysis
    md.append("## 9. Correlation Analysis")
    md.append("> ⚠️ **Scientific Caveat:** *Correlation measures statistical association, not causal dependency.* "
              "A high correlation coefficient between two variables does not prove that one drives or produces the other.\n")

    pos_df = strong_corrs["positive"]
    neg_df = strong_corrs["negative"]

    if not pos_df.empty:
        md.append("### Top Positive Correlations")
        md.append("| Feature 1 | Feature 2 | Pearson (r) | Strength |")
        md.append("| :--- | :--- | :--- | :--- |")
        for _, r in pos_df.iterrows():
            md.append(f"| **{r['Feature 1']}** | **{r['Feature 2']}** | {r['Correlation']} | {r['Strength']} |")
        md.append("")

    if not neg_df.empty:
        md.append("### Top Negative Correlations")
        md.append("| Feature 1 | Feature 2 | Pearson (r) | Strength |")
        md.append("| :--- | :--- | :--- | :--- |")
        for _, r in neg_df.iterrows():
            md.append(f"| **{r['Feature 1']}** | **{r['Feature 2']}** | {r['Correlation']} | {r['Strength']} |")
        md.append("")

    if target_col and target_col in df_cleaned.columns:
        md.append(f"### Target Association Inquiry: `{target_col}`")
        md.append(f"The user selected `{target_col}` as the primary focal variable for relationship modeling. "
                  "Key associations identified in bivariate comparisons reflect the core drivers of variance in this outcome.\n")

    # 10. Key Findings
    md.append("## 10. Key Findings")
    md.append("*Computed Observations (Preliminary Findings):*\n")
    if key_findings:
        for idx, f in enumerate(key_findings, 1):
            md.append(f"**{idx}. [{f['category']}] {f['title']}**  \n{f['description']}\n")
    else:
        md.append("No anomalous or extreme findings were automatically isolated.\n")

    # 11. Limitations
    md.append("## 11. Limitations")
    md.append(
        "1. **Observational Nature:** Data reflects non-experimental observation, meaning unmeasured confounding variables may explain apparent associations.\n"
        "2. **Information Loss via Missingness:** Substantial missingness in certain attributes (e.g. historical records) can bias sample subsets unless systematically imputed.\n"
        "3. **Linearity Assumptions:** Pearson correlation measures purely linear association; non-linear dependencies may register near zero.\n"
        "4. **Outlier Sensitivity:** Mean and standard deviation are sensitive to extreme observations; median and IQR provide complementary stability.\n"
    )

    # 12. Conclusion
    md.append("## 12. Conclusion")
    md.append(
        f"This exploratory data analysis of **{dataset_name}** successfully characterized the fundamental "
        f"properties of its {overview_clean['total_rows']:,} records across {overview_clean['total_columns']} features. "
        "The systematic combination of descriptive summaries, outlier auditing, bivariate comparisons, "
        "and correlation mapping highlights clear structural patterns while reinforcing the analytical discipline "
        "of validating hypotheses before inferring causation.\n"
    )

    # 13. Future Improvements
    md.append("## 13. Future Improvements")
    md.append(
        "1. **Advanced Imputation:** Experiment with model-based imputation techniques (e.g. K-Nearest Neighbors or MissForest) for continuous features.\n"
        "2. **Feature Engineering:** Derive composite indicators (e.g. interaction terms, ratio features, family size aggregates) to enhance signal.\n"
        "3. **Supervised Modeling:** Train predictive classifiers or regressors (e.g. Logistic Regression, Random Forests) with cross-validation.\n"
        "4. **Hypothesis Testing:** Formalize parametric and non-parametric hypothesis tests (Mann-Whitney U, Kruskal-Wallis) for deeper group comparisons.\n"
    )

    return "\n".join(md)
