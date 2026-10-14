"""
Data Visualization Module for Exploratory Data Analysis (EDA).

Generates modern, interactive Plotly visualizations and aesthetic Seaborn plots:
- Histograms with KDE / marginal box plots
- Boxplots with IQR whiskers and optional category grouping
- Categorical frequency bar charts
- Interactive annotated correlation heatmaps
- Scatter plots with OLS regression trendlines
- Bivariate categorical cross-comparison bar charts
- Multi-variable pair plots via Seaborn
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Consistent color themes
PRIMARY_COLOR = "#1f77b4"
PALETTE = px.colors.qualitative.Plotly


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive bar chart of missing value counts and percentages.
    """
    missing_series = df.isna().sum()
    missing_cols = missing_series[missing_series > 0].sort_values(ascending=False)

    if missing_cols.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Clean Dataset: Zero Missing Values Detected!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#2ca02c")
        )
        fig.update_layout(
            title="Missing Value Inspection",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
            height=300
        )
        return fig

    total_rows = len(df)
    missing_pct = ((missing_cols / total_rows) * 100).round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=missing_cols.index,
        y=missing_cols.values,
        text=[f"{count:,} ({pct}%)" for count, pct in zip(missing_cols.values, missing_pct.values)],
        textposition="auto",
        marker=dict(
            color=missing_pct.values,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Missing %")
        ),
        hovertemplate="<b>%{x}</b><br>Missing Count: %{y:,}<br>Missing: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Missing Values by Column</b>",
        xaxis_title="Column Name",
        yaxis_title="Missing Count (Rows)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=50)
    )
    return fig


def plot_numerical_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Generates an interactive histogram with marginal boxplot and distribution curve.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    sub_df = df.dropna(subset=[column])
    if color_col and color_col in df.columns:
        fig = px.histogram(
            sub_df,
            x=column,
            color=color_col,
            nbins=bins,
            marginal="box",
            opacity=0.75,
            barmode="overlay",
            title=f"<b>Distribution of {column} (Grouped by {color_col})</b>",
            template="plotly_white"
        )
    else:
        fig = px.histogram(
            sub_df,
            x=column,
            nbins=bins,
            marginal="box",
            title=f"<b>Distribution of {column}</b>",
            color_discrete_sequence=[PRIMARY_COLOR],
            template="plotly_white"
        )

    # Compute key stats for annotation
    mean_val = sub_df[column].mean()
    median_val = sub_df[column].median()

    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="#d62728",
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top left"
    )
    fig.add_vline(
        x=median_val, line_dash="dot", line_color="#2ca02c",
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency / Count",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    group_by: Optional[str] = None
) -> go.Figure:
    """
    Generates a boxplot showing quartiles, median, and statistical outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    sub_df = df.dropna(subset=[column])
    if group_by and group_by in df.columns:
        fig = px.box(
            sub_df,
            x=group_by,
            y=column,
            color=group_by,
            points="outliers",
            title=f"<b>Boxplot of {column} across {group_by}</b>",
            template="plotly_white"
        )
        fig.update_layout(xaxis_title=group_by, yaxis_title=column)
    else:
        fig = px.box(
            sub_df,
            y=column,
            points="outliers",
            title=f"<b>Boxplot of {column} (Outlier Detection)</b>",
            color_discrete_sequence=[PRIMARY_COLOR],
            template="plotly_white"
        )
        fig.update_layout(yaxis_title=column)

    fig.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    top_n: int = 15
) -> go.Figure:
    """
    Renders an interactive bar chart displaying categorical counts and frequencies.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    counts = df[column].value_counts(dropna=False).head(top_n)
    total = len(df)
    labels = [str(k) if pd.notna(k) else "<Missing>" for k in counts.index]
    percentages = ((counts.values / total) * 100).round(1)

    fig = go.Figure(go.Bar(
        x=labels,
        y=counts.values,
        text=[f"{c:,} ({pct}%)" for c, pct in zip(counts.values, percentages)],
        textposition="auto",
        marker=dict(color=PALETTE[:len(labels)]),
        hovertemplate="<b>Category:</b> %{x}<br><b>Count:</b> %{y:,}<br><b>Percent:</b> %{text}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>Category Frequencies: {column}</b> (Top {len(labels)})",
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=60, b=50)
    )
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix Heatmap"
) -> go.Figure:
    """
    Renders an interactive correlation heatmap with cell value annotations.
    """
    if corr_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient Numerical Columns for Correlation Matrix (Need >= 2).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, template="plotly_white", height=350)
        return fig

    z_values = corr_matrix.values
    x_labels = corr_matrix.columns.tolist()
    y_labels = corr_matrix.index.tolist()

    text_annotations = [[f"{val:.2f}" for val in row] for row in z_values]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Correlation (r)"),
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"<b>{title}</b>",
        template="plotly_white",
        height=max(450, len(x_labels) * 45),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    add_trendline: bool = True
) -> go.Figure:
    """
    Generates an interactive scatter plot with optional regression trendline.
    Computes linear regression using scipy.stats without external dependencies.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Selected columns not in DataFrame.")

    sub_df = df.dropna(subset=[x_col, y_col])

    if color_col and color_col in df.columns:
        fig = px.scatter(
            sub_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"<b>{y_col} vs {x_col} (Colored by {color_col})</b>",
            template="plotly_white",
            opacity=0.75
        )
    else:
        fig = px.scatter(
            sub_df,
            x=x_col,
            y=y_col,
            title=f"<b>Relationship: {y_col} vs {x_col}</b>",
            color_discrete_sequence=[PRIMARY_COLOR],
            template="plotly_white",
            opacity=0.75
        )

    # Add OLS regression trendline if requested
    if add_trendline and len(sub_df) >= 2:
        try:
            x_vals = pd.to_numeric(sub_df[x_col], errors="coerce")
            y_vals = pd.to_numeric(sub_df[y_col], errors="coerce")
            valid_mask = x_vals.notna() & y_vals.notna()
            if valid_mask.sum() >= 2:
                slope, intercept, r_val, p_val, _ = stats.linregress(
                    x_vals[valid_mask], y_vals[valid_mask]
                )
                x_sorted = np.linspace(x_vals[valid_mask].min(), x_vals[valid_mask].max(), 100)
                y_pred = slope * x_sorted + intercept

                fig.add_trace(go.Scatter(
                    x=x_sorted,
                    y=y_pred,
                    mode="lines",
                    name=f"OLS Fit (r={r_val:.2f}, R²={r_val**2:.2f})",
                    line=dict(color="#d62728", dash="dash", width=2.5),
                    hoverinfo="skip"
                ))
        except Exception:
            pass  # Fallback gracefully if columns are non-numeric

    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=480,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_bivariate_categorical(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    as_percentage: bool = True
) -> go.Figure:
    """
    Generates a grouped or 100% stacked bar chart comparing two categorical features.
    """
    sub_df = df.dropna(subset=[feature_col, target_col])
    if len(sub_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No matching data points.", x=0.5, y=0.5, showarrow=False)
        return fig

    if as_percentage:
        ct = pd.crosstab(sub_df[feature_col], sub_df[target_col], normalize="index") * 100
        y_title = "Percentage within Category (%)"
        barmode = "group"
    else:
        ct = pd.crosstab(sub_df[feature_col], sub_df[target_col])
        y_title = "Record Count"
        barmode = "group"

    fig = go.Figure()
    for col in ct.columns:
        fig.add_trace(go.Bar(
            name=f"{target_col}: {col}",
            x=ct.index.astype(str),
            y=ct[col].values,
            text=[f"{v:.1f}%" if as_percentage else f"{int(v):,}" for v in ct[col].values],
            textposition="auto"
        ))

    fig.update_layout(
        barmode=barmode,
        title=f"<b>{feature_col} vs {target_col} Distribution</b>",
        xaxis_title=feature_col,
        yaxis_title=y_title,
        template="plotly_white",
        height=450,
        legend_title=target_col,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    columns: List[str],
    hue_col: Optional[str] = None,
    max_samples: int = 500
) -> plt.Figure:
    """
    Generates a static pairplot figure using Seaborn for multi-variable distribution inspection.
    """
    sub_cols = list(columns)
    if hue_col and hue_col in df.columns and hue_col not in sub_cols:
        sub_cols.append(hue_col)

    sample_df = df[sub_cols].dropna()
    if len(sample_df) > max_samples:
        sample_df = sample_df.sample(max_samples, random_state=42)

    sns.set_theme(style="whitegrid")
    g = sns.pairplot(
        sample_df,
        vars=columns,
        hue=hue_col if (hue_col and hue_col in sample_df.columns) else None,
        diag_kind="kde",
        corner=False,
        plot_kws={"alpha": 0.6, "s": 30}
    )
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle("Pairwise Relationships Matrix", fontsize=14, fontweight="bold")
    return g.fig
