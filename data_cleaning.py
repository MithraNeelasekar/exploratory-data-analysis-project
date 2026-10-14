"""
Data Cleaning and Preparation Module for Exploratory Data Analysis (EDA).

This module provides functional, reproducible data cleaning operations:
- Missing value detection and imputation (mean, median, mode, constant, drop)
- Duplicate row detection and removal
- Outlier detection and handling using the Interquartile Range (IQR) method
- Column data type inspection and casting
- Before-and-after dataset comparison audit
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a comprehensive summary of missing values for all columns.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DataFrame with columns: Column, Data Type, Missing Count, Missing Percentage (%), Non-Null Count.
    """
    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame(columns=["Column", "Data Type", "Missing Count", "Missing (%)", "Non-Null Count"])

    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / total_rows) * 100
    dtypes = df.dtypes

    summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtypes[col]) for col in df.columns],
        "Missing Count": missing_counts.values,
        "Missing (%)": missing_pct.round(2).values,
        "Non-Null Count": (total_rows - missing_counts).values
    })

    return summary.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)


def get_duplicate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies duplicate rows in the DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dictionary containing duplicate count, duplicate percentage, and sample duplicate rows.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "has_duplicates": False,
            "duplicate_indices": []
        }

    duplicate_mask = df.duplicated(keep="first")
    dup_count = int(duplicate_mask.sum())
    dup_pct = round((dup_count / total_rows) * 100, 2)

    return {
        "duplicate_count": dup_count,
        "duplicate_pct": dup_pct,
        "has_duplicates": dup_count > 0,
        "duplicate_indices": df.index[duplicate_mask].tolist()
    }


def remove_duplicates(df: pd.DataFrame, keep: str = "first") -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate records from the DataFrame while keeping the original intact.

    Args:
        df: Input pandas DataFrame.
        keep: Which duplicate record to retain ('first', 'last', or False to drop all).

    Returns:
        Tuple of (cleaned DataFrame copy, number of rows removed).
    """
    before_count = len(df)
    cleaned_df = df.drop_duplicates(keep=keep).copy()
    rows_removed = before_count - len(cleaned_df)
    return cleaned_df, rows_removed


def impute_numerical_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    custom_value: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes missing values in a numerical column using the specified strategy.

    Args:
        df: Input pandas DataFrame.
        column: Column name to impute.
        strategy: Strategy name ('mean', 'median', 'mode', 'constant', 'drop').
        custom_value: Value used when strategy is 'constant'.

    Returns:
        Tuple of (new DataFrame with imputation, audit details dict).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame.")

    cleaned_df = df.copy()
    missing_before = int(cleaned_df[column].isna().sum())

    if missing_before == 0:
        return cleaned_df, {
            "column": column,
            "strategy": strategy,
            "imputed_count": 0,
            "fill_value": None,
            "description": f"No missing values in '{column}'."
        }

    fill_value: Any = None
    if strategy == "mean":
        fill_value = cleaned_df[column].mean()
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    elif strategy == "median":
        fill_value = cleaned_df[column].median()
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    elif strategy == "mode":
        mode_series = cleaned_df[column].mode()
        fill_value = mode_series.iloc[0] if not mode_series.empty else 0
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    elif strategy == "constant":
        fill_value = 0.0 if custom_value is None else custom_value
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    elif strategy == "drop":
        cleaned_df = cleaned_df.dropna(subset=[column]).copy()
        fill_value = "DROPPED_ROWS"
    else:
        raise ValueError(f"Unsupported strategy '{strategy}' for numerical column.")

    return cleaned_df, {
        "column": column,
        "strategy": strategy,
        "imputed_count": missing_before,
        "fill_value": fill_value,
        "description": f"Applied {strategy} imputation on '{column}' ({missing_before} cells affected)."
    }


def impute_categorical_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mode",
    custom_value: Optional[str] = "Missing"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputes missing values in a categorical column.

    Args:
        df: Input pandas DataFrame.
        column: Column name to impute.
        strategy: Strategy name ('mode', 'constant', 'drop').
        custom_value: String to fill when strategy is 'constant'.

    Returns:
        Tuple of (new DataFrame with imputation, audit details dict).
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in DataFrame.")

    cleaned_df = df.copy()
    missing_before = int(cleaned_df[column].isna().sum())

    if missing_before == 0:
        return cleaned_df, {
            "column": column,
            "strategy": strategy,
            "imputed_count": 0,
            "fill_value": None,
            "description": f"No missing values in '{column}'."
        }

    fill_value: Any = None
    if strategy == "mode":
        mode_series = cleaned_df[column].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    elif strategy == "constant":
        fill_value = "Missing" if custom_value is None else str(custom_value)
        cleaned_df[column] = cleaned_df[column].fillna(fill_value)
    elif strategy == "drop":
        cleaned_df = cleaned_df.dropna(subset=[column]).copy()
        fill_value = "DROPPED_ROWS"
    else:
        raise ValueError(f"Unsupported strategy '{strategy}' for categorical column.")

    return cleaned_df, {
        "column": column,
        "strategy": strategy,
        "imputed_count": missing_before,
        "fill_value": fill_value,
        "description": f"Applied {strategy} imputation on '{column}' ({missing_before} cells affected)."
    }


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Dict[str, Any]:
    """
    Detects statistical outliers in a numerical column using the Interquartile Range (IQR) method.

    Rule:
        IQR = Q3 - Q1
        Lower Bound = Q1 - (multiplier * IQR)
        Upper Bound = Q3 + (multiplier * IQR)
        Outlier if value < Lower Bound or value > Upper Bound.

    Args:
        df: Input pandas DataFrame.
        column: Numerical column name.
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Dictionary with Q1, Q3, IQR, bounds, outlier count, percentage, and summary.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    series = df[column].dropna()
    if len(series) == 0:
        return {
            "column": column,
            "count": 0,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "min_val": None,
            "max_val": None
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round((outlier_count / len(series)) * 100, 2)

    return {
        "column": column,
        "count": len(series),
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "min_val": round(float(series.min()), 3),
        "max_val": round(float(series.max()), 3),
        "outlier_indices": df.index[df[column].isin(series[outlier_mask])].tolist()
    }


def handle_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    action: str = "clip",
    multiplier: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles outliers detected via the IQR method.

    Args:
        df: Input pandas DataFrame.
        column: Numerical column name.
        action: 'clip' (cap/floor at bounds) or 'drop' (remove outlier records).
        multiplier: IQR multiplier (default 1.5).

    Returns:
        Tuple of (modified DataFrame copy, audit report dict).
    """
    info = detect_outliers_iqr(df, column, multiplier=multiplier)
    if info["outlier_count"] == 0:
        return df.copy(), {
            "column": column,
            "action": action,
            "affected_rows": 0,
            "message": f"No outliers detected in '{column}' using {multiplier}x IQR."
        }

    cleaned_df = df.copy()
    lower_b = info["lower_bound"]
    upper_b = info["upper_bound"]

    if action == "clip":
        cleaned_df[column] = cleaned_df[column].clip(lower=lower_b, upper=upper_b)
        affected = info["outlier_count"]
        msg = f"Capped {affected} outlier values in '{column}' to range [{lower_b}, {upper_b}]."
    elif action == "drop":
        initial_len = len(cleaned_df)
        mask = (cleaned_df[column].isna()) | ((cleaned_df[column] >= lower_b) & (cleaned_df[column] <= upper_b))
        cleaned_df = cleaned_df[mask].copy()
        affected = initial_len - len(cleaned_df)
        msg = f"Dropped {affected} rows with outliers in '{column}' outside [{lower_b}, {upper_b}]."
    else:
        raise ValueError(f"Unsupported outlier action '{action}'. Choose 'clip' or 'drop'.")

    return cleaned_df, {
        "column": column,
        "action": action,
        "affected_rows": affected,
        "message": msg
    }


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Converts a column in the DataFrame to a specified target data type safely.

    Args:
        df: Input DataFrame.
        column: Column name.
        target_type: 'numeric', 'integer', 'string', 'category', or 'datetime'.

    Returns:
        Tuple of (converted DataFrame, success boolean, status message).
    """
    if column not in df.columns:
        return df.copy(), False, f"Column '{column}' not found."

    cleaned_df = df.copy()
    try:
        if target_type == "numeric":
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
        elif target_type == "integer":
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce").round().astype("Int64")
        elif target_type == "string":
            cleaned_df[column] = cleaned_df[column].astype(str)
        elif target_type == "category":
            cleaned_df[column] = cleaned_df[column].astype("category")
        elif target_type == "datetime":
            cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors="coerce")
        else:
            return df.copy(), False, f"Unsupported target type '{target_type}'."

        return cleaned_df, True, f"Successfully converted '{column}' to {target_type}."
    except Exception as e:
        return df.copy(), False, f"Failed converting '{column}' to {target_type}: {str(e)}"


def get_cleaning_diff(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """
    Compares the original and cleaned DataFrames to generate an audit diff.

    Args:
        df_before: DataFrame prior to cleaning.
        df_after: DataFrame after cleaning.

    Returns:
        Dictionary with comparative metrics: rows, missing cells, duplicates, memory.
    """
    rows_before = len(df_before)
    rows_after = len(df_after)
    cols_before = len(df_before.columns)
    cols_after = len(df_after.columns)

    missing_before = int(df_before.isna().sum().sum())
    missing_after = int(df_after.isna().sum().sum())

    dups_before = int(df_before.duplicated().sum())
    dups_after = int(df_after.duplicated().sum())

    mem_before_kb = round(df_before.memory_usage(deep=True).sum() / 1024, 2)
    mem_after_kb = round(df_after.memory_usage(deep=True).sum() / 1024, 2)

    return {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_diff": rows_after - rows_before,
        "cols_before": cols_before,
        "cols_after": cols_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "missing_diff": missing_after - missing_before,
        "duplicates_before": dups_before,
        "duplicates_after": dups_after,
        "duplicates_diff": dups_after - dups_before,
        "memory_before_kb": mem_before_kb,
        "memory_after_kb": mem_after_kb
    }
