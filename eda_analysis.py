"""
Exploratory Data Analysis (EDA) Statistical Analysis Module.

Provides comprehensive descriptive statistics, distribution metrics,
correlation modeling, bivariate target interactions, and automated preliminary findings.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from data_cleaning import (
    get_missing_summary,
    get_duplicate_summary,
    detect_outliers_iqr
)


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Separates DataFrame columns into numerical, categorical, and datetime groups.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dictionary with keys 'numerical', 'categorical', 'datetime'.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    # Categorical includes object, category, bool, and string
    categorical_cols = [
        col for col in df.columns
        if col not in numerical_cols and col not in datetime_cols
    ]
    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols
    }


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates high-level metadata and dataset dimensions.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dictionary of overview statistics.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows * 100), 2) if total_rows > 0 else 0.0
    mem_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)

    col_types = get_column_types(df)

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_percentage": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": duplicate_pct,
        "memory_usage_mb": mem_usage_mb,
        "numerical_columns": col_types["numerical"],
        "categorical_columns": col_types["categorical"],
        "datetime_columns": col_types["datetime"],
        "num_count": len(col_types["numerical"]),
        "cat_count": len(col_types["categorical"])
    }


def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates rich descriptive statistics for numerical columns:
    mean, median, mode, min, max, std, variance, quartiles, IQR, skewness, kurtosis.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame indexed by column name with descriptive metrics.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return pd.DataFrame()

    records = []
    for col in num_cols:
        series = df[col].dropna()
        n_missing = int(df[col].isna().sum())
        missing_pct = round((n_missing / len(df)) * 100, 2) if len(df) > 0 else 0.0

        if len(series) == 0:
            records.append({
                "Column": col,
                "Count": 0,
                "Missing": n_missing,
                "Missing (%)": missing_pct,
                "Mean": np.nan,
                "Std Dev": np.nan,
                "Variance": np.nan,
                "Min": np.nan,
                "25% (Q1)": np.nan,
                "50% (Median)": np.nan,
                "75% (Q3)": np.nan,
                "Max": np.nan,
                "Range": np.nan,
                "IQR": np.nan,
                "Mode": np.nan,
                "Skewness": np.nan,
                "Kurtosis": np.nan
            })
            continue

        mean_val = series.mean()
        std_val = series.std(ddof=1) if len(series) > 1 else 0.0
        var_val = series.var(ddof=1) if len(series) > 1 else 0.0
        min_val = series.min()
        q1 = series.quantile(0.25)
        median_val = series.median()
        q3 = series.quantile(0.75)
        max_val = series.max()
        range_val = max_val - min_val
        iqr_val = q3 - q1

        # Mode calculation
        mode_series = series.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan

        # Skewness & Kurtosis (Fisher-Pearson)
        skew_val = series.skew() if len(series) > 2 else 0.0
        kurt_val = series.kurt() if len(series) > 3 else 0.0

        records.append({
            "Column": col,
            "Count": len(series),
            "Missing": n_missing,
            "Missing (%)": missing_pct,
            "Mean": round(float(mean_val), 3),
            "Std Dev": round(float(std_val), 3),
            "Variance": round(float(var_val), 3),
            "Min": round(float(min_val), 3),
            "25% (Q1)": round(float(q1), 3),
            "50% (Median)": round(float(median_val), 3),
            "75% (Q3)": round(float(q3), 3),
            "Max": round(float(max_val), 3),
            "Range": round(float(range_val), 3),
            "IQR": round(float(iqr_val), 3),
            "Mode": round(float(mode_val), 3) if not np.isnan(mode_val) else np.nan,
            "Skewness": round(float(skew_val), 3),
            "Kurtosis": round(float(kurt_val), 3)
        })

    return pd.DataFrame(records)


def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistical summaries for non-numeric (categorical, boolean, string) columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with count, unique, mode, frequency of top value, and missing percentages.
    """
    cat_cols = get_column_types(df)["categorical"]
    if not cat_cols:
        return pd.DataFrame()

    records = []
    total_rows = len(df)
    for col in cat_cols:
        series = df[col]
        non_null_count = int(series.notna().sum())
        n_missing = int(series.isna().sum())
        missing_pct = round((n_missing / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        value_counts = series.value_counts(dropna=True)
        if not value_counts.empty:
            top_val = value_counts.index[0]
            top_freq = int(value_counts.iloc[0])
            top_pct = round((top_freq / non_null_count) * 100, 2) if non_null_count > 0 else 0.0
        else:
            top_val = "N/A"
            top_freq = 0
            top_pct = 0.0

        records.append({
            "Column": col,
            "Total Count": total_rows,
            "Non-Null Count": non_null_count,
            "Missing": n_missing,
            "Missing (%)": missing_pct,
            "Unique Values": unique_count,
            "Mode (Most Frequent)": str(top_val),
            "Mode Frequency": top_freq,
            "Mode (%)": top_pct
        })

    return pd.DataFrame(records)


def get_frequency_distribution(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Calculates detailed frequency counts and percentages for a specific column.

    Args:
        df: Input DataFrame.
        column: Categorical column name.
        top_n: Maximum number of distinct categories to return.

    Returns:
        DataFrame with Category, Count, Percentage (%), and Cumulative Percentage (%).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    counts = df[column].value_counts(dropna=False)
    total = len(df)

    dist = pd.DataFrame({
        "Category": [str(c) if pd.notna(c) else "<Missing / NaN>" for c in counts.index],
        "Count": counts.values,
        "Percentage (%)": ((counts.values / total) * 100).round(2)
    })

    if top_n and len(dist) > top_n:
        top_part = dist.iloc[:top_n].copy()
        other_count = dist.iloc[top_n:]["Count"].sum()
        other_pct = round((other_count / total) * 100, 2)
        other_row = pd.DataFrame([{
            "Category": f"Other ({len(dist) - top_n} categories)",
            "Count": other_count,
            "Percentage (%)": other_pct
        }])
        dist = pd.concat([top_part, other_row], ignore_index=True)

    dist["Cumulative (%)"] = dist["Percentage (%)"].cumsum().round(2)
    return dist


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson"
) -> pd.DataFrame:
    """
    Computes pairwise correlation matrix for numerical columns.

    Args:
        df: Input DataFrame.
        method: 'pearson' (linear) or 'spearman' (rank monotonic).

    Returns:
        Correlation matrix DataFrame.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    corr = num_df.corr(method=method)
    return corr.round(3)


def get_strongest_correlations(
    corr_matrix: pd.DataFrame,
    top_n: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Extracts the strongest positive and negative correlation pairs from a correlation matrix.

    Args:
        corr_matrix: Pairwise correlation matrix.
        top_n: Number of top pairs to retrieve.

    Returns:
        Dictionary with 'positive' and 'negative' DataFrames.
    """
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        empty_df = pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation", "Strength"])
        return {"positive": empty_df, "negative": empty_df}

    pairs = []
    cols = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = corr_matrix.loc[col1, col2]
            if pd.notna(val):
                pairs.append({
                    "Feature 1": col1,
                    "Feature 2": col2,
                    "Correlation": float(val),
                    "AbsCorrelation": abs(float(val))
                })

    if not pairs:
        empty_df = pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation", "Strength"])
        return {"positive": empty_df, "negative": empty_df}

    pairs_df = pd.DataFrame(pairs)

    def describe_strength(r: float) -> str:
        abs_r = abs(r)
        if abs_r >= 0.7:
            return "Strong"
        elif abs_r >= 0.4:
            return "Moderate"
        elif abs_r >= 0.2:
            return "Weak"
        else:
            return "Very Weak / None"

    pairs_df["Strength"] = pairs_df["Correlation"].apply(describe_strength)

    # Positive correlations (descending)
    pos_df = pairs_df[pairs_df["Correlation"] > 0].sort_values(
        by="Correlation", ascending=False
    ).head(top_n)[["Feature 1", "Feature 2", "Correlation", "Strength"]].reset_index(drop=True)

    # Negative correlations (ascending)
    neg_df = pairs_df[pairs_df["Correlation"] < 0].sort_values(
        by="Correlation", ascending=True
    ).head(top_n)[["Feature 1", "Feature 2", "Correlation", "Strength"]].reset_index(drop=True)

    return {
        "positive": pos_df,
        "negative": neg_df
    }


def analyze_target_relationship(
    df: pd.DataFrame,
    target_col: str,
    feature_col: str
) -> Dict[str, Any]:
    """
    Examines the statistical relationship between an outcome/target column and a feature column.
    Dynamically routes according to variable types:
    - Numerical vs Numerical: Pearson correlation, Spearman rank, linear regression slope.
    - Categorical Target vs Numerical Feature: Group means, medians, and ANOVA / t-test.
    - Categorical Target vs Categorical Feature: Cross-tabulation, Chi-square test, Cramer's V.

    Args:
        df: Input DataFrame.
        target_col: Target or outcome variable.
        feature_col: Explanatory or predictor variable.

    Returns:
        Dictionary with statistical tests, summaries, and interpretation.
    """
    if target_col not in df.columns or feature_col not in df.columns:
        raise ValueError("Target or feature column not found in dataset.")

    sub_df = df[[target_col, feature_col]].dropna()
    if len(sub_df) < 3:
        return {
            "type": "insufficient_data",
            "message": "Not enough non-missing data points to evaluate relationship."
        }

    is_target_num = pd.api.types.is_numeric_dtype(df[target_col])
    is_feature_num = pd.api.types.is_numeric_dtype(df[feature_col])

    # Treat binary 0/1 columns (like Survived) optionally as categorical when desired
    target_nunique = df[target_col].nunique()

    # Case 1: Numerical vs Numerical
    if is_target_num and is_feature_num and target_nunique > 5:
        p_corr, p_val = stats.pearsonr(sub_df[feature_col], sub_df[target_col])
        s_corr, s_pval = stats.spearmanr(sub_df[feature_col], sub_df[target_col])
        slope, intercept, r_value, r_pval, std_err = stats.linregress(
            sub_df[feature_col], sub_df[target_col]
        )

        return {
            "type": "num_vs_num",
            "feature": feature_col,
            "target": target_col,
            "pearson_r": round(float(p_corr), 3),
            "pearson_p_value": float(f"{p_val:.2e}"),
            "spearman_rho": round(float(s_corr), 3),
            "spearman_p_value": float(f"{s_pval:.2e}"),
            "r_squared": round(float(r_value ** 2), 3),
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "interpretation": (
                f"Pearson r = {round(p_corr, 3)} indicates a "
                f"{'positive' if p_corr > 0 else 'negative'} linear correlation. "
                f"R² = {round(r_value**2, 3)} implies that approximately "
                f"{round(r_value**2 * 100, 1)}% of variance in '{target_col}' is explained by '{feature_col}'."
            )
        }

    # Case 2: Categorical Target (e.g. Survived) vs Numerical Feature (e.g. Age, Fare)
    elif (not is_target_num or target_nunique <= 10) and is_feature_num:
        groups = [group[feature_col].values for _, group in sub_df.groupby(target_col)]
        summary_by_group = sub_df.groupby(target_col)[feature_col].agg(
            Count="count", Mean="mean", Median="median", Std="std"
        ).round(3).reset_index()

        f_stat, f_pval = (np.nan, np.nan)
        if len(groups) >= 2 and all(len(g) > 1 for g in groups):
            try:
                f_stat, f_pval = stats.f_oneway(*groups)
            except Exception:
                pass

        return {
            "type": "cat_target_vs_num_feature",
            "feature": feature_col,
            "target": target_col,
            "group_table": summary_by_group,
            "f_statistic": round(float(f_stat), 3) if pd.notna(f_stat) else None,
            "p_value": float(f"{f_pval:.2e}") if pd.notna(f_pval) else None,
            "interpretation": (
                f"Numerical distributions of '{feature_col}' across '{target_col}' classes. "
                f"ANOVA p-value = {f'{f_pval:.2e}' if pd.notna(f_pval) else 'N/A'}, suggesting "
                f"{'statistically significant differences between groups' if (pd.notna(f_pval) and f_pval < 0.05) else 'no significant difference detected'}."
            )
        }

    # Case 3: Both Categorical (or Categorical Feature with low-unique target)
    else:
        contingency = pd.crosstab(sub_df[feature_col], sub_df[target_col])
        contingency_pct = pd.crosstab(sub_df[feature_col], sub_df[target_col], normalize="index") * 100

        try:
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
        except Exception:
            chi2, p, dof, cramers_v = (np.nan, np.nan, np.nan, np.nan)

        return {
            "type": "cat_vs_cat",
            "feature": feature_col,
            "target": target_col,
            "contingency_table": contingency,
            "contingency_pct_table": contingency_pct.round(2),
            "chi2": round(float(chi2), 3) if pd.notna(chi2) else None,
            "p_value": float(f"{p:.2e}") if pd.notna(p) else None,
            "dof": int(dof) if pd.notna(dof) else None,
            "cramers_v": round(float(cramers_v), 3) if pd.notna(cramers_v) else None,
            "interpretation": (
                f"Contingency analysis between '{feature_col}' and '{target_col}'. "
                f"Chi-square p-value = {f'{p:.2e}' if pd.notna(p) else 'N/A'} "
                f"(Cramer's V = {round(cramers_v, 3) if pd.notna(cramers_v) else 'N/A'}), indicating "
                f"{'a statistically significant dependency' if (pd.notna(p) and p < 0.05) else 'no significant association'}."
            )
        }


def generate_key_findings(df: pd.DataFrame, target_col: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Inspects the actual dataset and computes structured analytical observations.
    Never invents or fabricates data.

    Args:
        df: Input DataFrame.
        target_col: Optional chosen target column.

    Returns:
        List of dictionaries with 'title', 'category', 'description', and 'type'.
    """
    findings: List[Dict[str, str]] = []
    total_rows = len(df)
    if total_rows == 0:
        return findings

    # 1. Dataset Dimensions & Completeness
    missing_summary = get_missing_summary(df)
    missing_cols = missing_summary[missing_summary["Missing Count"] > 0]
    if not missing_cols.empty:
        top_missing = missing_cols.iloc[0]
        findings.append({
            "category": "Data Quality",
            "title": "Missing Value Concentrations",
            "description": (
                f"The dataset contains missing entries across {len(missing_cols)} column(s). "
                f"'{top_missing['Column']}' has the highest missing count ({int(top_missing['Missing Count'])} entries, "
                f"{top_missing['Missing (%)']}% of the dataset), which requires deliberate handling."
            )
        })
    else:
        findings.append({
            "category": "Data Quality",
            "title": "Clean Completeness",
            "description": "Zero missing values were detected in any column of the dataset."
        })

    # 2. Duplicate Records
    dup_info = get_duplicate_summary(df)
    if dup_info["has_duplicates"]:
        findings.append({
            "category": "Data Integrity",
            "title": "Duplicate Rows Present",
            "description": (
                f"{dup_info['duplicate_count']} duplicate row(s) ({dup_info['duplicate_pct']}%) were detected. "
                "Review whether duplicate records represent repeat observations or redundant logging."
            )
        })

    # 3. Skewness in Numerical Data
    num_summary = get_numerical_summary(df)
    if not num_summary.empty and "Skewness" in num_summary.columns:
        high_skew = num_summary[num_summary["Skewness"].abs() > 1.5]
        if not high_skew.empty:
            skewed_names = high_skew["Column"].tolist()
            findings.append({
                "category": "Distribution Analysis",
                "title": "Pronounced Asymmetry / Skewness",
                "description": (
                    f"Columns {skewed_names} exhibit high skewness (|skew| > 1.5), "
                    "indicating long-tailed distributions where the median is a more robust measure of central tendency than the mean."
                )
            })

    # 4. Outliers via IQR
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = []
    for col in num_cols:
        res = detect_outliers_iqr(df, col)
        if res["outlier_count"] > 0:
            outlier_cols.append(f"{col} ({res['outlier_count']} outliers, {res['outlier_pct']}%)")

    if outlier_cols:
        findings.append({
            "category": "Anomaly Detection",
            "title": "Statistical Outliers Identified",
            "description": (
                f"Outliers were detected via the 1.5x IQR rule in: {', '.join(outlier_cols[:4])}. "
                "These extreme values can inflate standard deviations and skew linear models if unaddressed."
            )
        })

    # 5. Correlation Patterns
    corr_matrix = compute_correlation_matrix(df, method="pearson")
    if not corr_matrix.empty:
        strong_pairs = get_strongest_correlations(corr_matrix, top_n=2)
        pos = strong_pairs["positive"]
        neg = strong_pairs["negative"]

        if not pos.empty and pos.iloc[0]["Correlation"] >= 0.4:
            top_pos = pos.iloc[0]
            findings.append({
                "category": "Correlation",
                "title": "Prominent Positive Correlation",
                "description": (
                    f"A notable positive relationship exists between '{top_pos['Feature 1']}' and '{top_pos['Feature 2']}' "
                    f"(Pearson r = {top_pos['Correlation']}, {top_pos['Strength']}). An increase in one tends to co-occur with an increase in the other."
                )
            })

        if not neg.empty and abs(neg.iloc[0]["Correlation"]) >= 0.4:
            top_neg = neg.iloc[0]
            findings.append({
                "category": "Correlation",
                "title": "Prominent Negative Correlation",
                "description": (
                    f"A notable inverse relationship exists between '{top_neg['Feature 1']}' and '{top_neg['Feature 2']}' "
                    f"(Pearson r = {top_neg['Correlation']}, {top_neg['Strength']}). As one increases, the other tends to decrease."
                )
            })

    # 6. Target Analysis (if specified)
    if target_col and target_col in df.columns:
        if df[target_col].nunique() <= 5:
            val_counts = df[target_col].value_counts(normalize=True) * 100
            classes_str = ", ".join([f"'{k}': {v:.1f}%" for k, v in val_counts.items()])
            findings.append({
                "category": "Target Variable",
                "title": f"Outcome Distribution for '{target_col}'",
                "description": (
                    f"Target column '{target_col}' has {df[target_col].nunique()} distinct classes ({classes_str}). "
                    "This baseline class distribution is essential context for evaluating influencing factors."
                )
            })

    return findings
