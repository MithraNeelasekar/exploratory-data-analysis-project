"""
Exploratory Data Analysis (EDA) Interactive Dashboard.

A modern, student- and faculty-friendly Streamlit web application
demonstrating comprehensive data exploration, cleaning, statistical analysis,
visualization, correlation analysis, and report generation.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

# Import project analytical modules
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


# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="EDA Project | Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, student-project presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .notice-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION & DATA LOADING
# ==============================================================================
def load_default_titanic() -> pd.DataFrame:
    """Loads bundled Titanic dataset with local fallback."""
    local_path = os.path.join(os.path.dirname(__file__), "data", "titanic.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    # Online fallback if file is ever moved
    import seaborn as sns
    df = sns.load_dataset("titanic")
    df.insert(0, "PassengerId", range(1, len(df) + 1))
    return df


if "df_original" not in st.session_state:
    st.session_state["df_original"] = load_default_titanic()
    st.session_state["df_clean"] = st.session_state["df_original"].copy()
    st.session_state["dataset_name"] = "Titanic Dataset (Default)"
    st.session_state["cleaning_history"] = []


# ==============================================================================
# SIDEBAR CONTROLS & NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("## 📊 **EDA Project**")
    st.caption("College Exploratory Data Analysis Dashboard")
    st.markdown("---")

    # Navigation menu
    menu = st.radio(
        "**Navigation**",
        [
            "🏠 Home",
            "🔍 Dataset Explorer",
            "🧹 Data Cleaning",
            "📈 Statistical Summary",
            "🎨 Visual Analysis",
            "🔗 Correlation Analysis",
            "📝 Insights and Report"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📂 **Dataset Source**")
    dataset_source = st.radio(
        "Choose Data Source:",
        ["Titanic Dataset (Default)", "Upload Custom CSV"],
        index=0 if "Titanic" in st.session_state["dataset_name"] else 1
    )

    if dataset_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                if len(uploaded_df) == 0:
                    st.error("Uploaded CSV is empty.")
                else:
                    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                        st.session_state["df_original"] = uploaded_df
                        st.session_state["df_clean"] = uploaded_df.copy()
                        st.session_state["dataset_name"] = uploaded_file.name
                        st.session_state["cleaning_history"] = []
                        st.session_state["last_uploaded_name"] = uploaded_file.name
                        st.success(f"Loaded '{uploaded_file.name}'!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        if st.session_state.get("dataset_name") != "Titanic Dataset (Default)":
            st.session_state["df_original"] = load_default_titanic()
            st.session_state["df_clean"] = st.session_state["df_original"].copy()
            st.session_state["dataset_name"] = "Titanic Dataset (Default)"
            st.session_state["cleaning_history"] = []
            st.session_state["last_uploaded_name"] = None

    st.markdown("---")
    st.markdown("### 🛠️ **Status**")
    active_rows = len(st.session_state["df_clean"])
    active_cols = len(st.session_state["df_clean"].columns)
    st.write(f"**Dataset:** `{st.session_state['dataset_name']}`")
    st.write(f"**Active Shape:** `{active_rows} rows × {active_cols} cols`")

    if st.button("🔄 Reset to Original Data", use_container_width=True):
        st.session_state["df_clean"] = st.session_state["df_original"].copy()
        st.session_state["cleaning_history"] = []
        st.rerun()


# Working references
df_original: pd.DataFrame = st.session_state["df_original"]
df_clean: pd.DataFrame = st.session_state["df_clean"]
dataset_name: str = st.session_state["dataset_name"]
cleaning_history = st.session_state["cleaning_history"]


# ==============================================================================
# SECTION 1: HOME
# ==============================================================================
if menu == "🏠 Home":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA) Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A complete, interactive analytical framework for statistical summarization, hypothesis testing, data cleaning, and visual exploration.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="notice-box">
        <strong>🎯 Project Objective:</strong> Analyze a structured dataset to uncover fundamental patterns, evaluate data hygiene,
        compute descriptive summaries, inspect correlations and key influencing factors, and present findings in an academic-grade structured report.
    </div>
    """, unsafe_allow_html=True)

    # 4 Quick Metric Cards
    overview = get_dataset_overview(df_clean)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Records</h4>
            <p>{overview['total_rows']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Variables (Features)</h4>
            <p>{overview['total_columns']}</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Missing Values</h4>
            <p>{overview['missing_percentage']}%</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Duplicate Rows</h4>
            <p>{overview['duplicate_rows']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🔬 **What is Exploratory Data Analysis (EDA)?**")
        st.markdown("""
        **Exploratory Data Analysis (EDA)** is an essential step in data science first formalized by statistician John Tukey.
        It emphasizes using statistical summaries and graphical representations to:
        1. **Understand Data Structure:** Examine feature types, scales, and distributions.
        2. **Audit Data Hygiene:** Identify missingness, duplicate observations, and formatting inconsistencies.
        3. **Spot Anomalies & Outliers:** Detect extreme points using the Interquartile Range (IQR) and boxplots.
        4. **Discover Relationships:** Map Pearson and Spearman correlations and cross-tabulate categories.
        5. **Formulate Hypotheses:** Generate research questions for future statistical modeling without jumping to false causal conclusions.
        """)

        st.markdown("### 🧭 **How to Navigate This Application**")
        st.markdown("""
        - **Dataset Explorer:** Review dataset dimensions, column data types, and inspect sample records.
        - **Data Cleaning:** Interactively impute missing values, handle duplicates, and cap statistical outliers.
        - **Statistical Summary:** Review comprehensive 5-number summaries, skewness, kurtosis, and category frequencies.
        - **Visual Analysis:** Explore interactive histograms, boxplots, scatter plots with OLS trendlines, and count plots.
        - **Correlation Analysis:** Inspect correlation heatmaps and evaluate relationship drivers for an outcome variable.
        - **Insights and Report:** Read dynamically generated findings and download the official 13-heading markdown report.
        """)

    with col_right:
        st.markdown("### 📋 **Active Dataset Profile**")
        st.info(f"**Dataset Name:** {dataset_name}\n\n"
                f"**Numerical Variables ({overview['num_count']}):**\n"
                f"`{', '.join(overview['numerical_columns'][:6])}{'...' if overview['num_count'] > 6 else ''}`\n\n"
                f"**Categorical Variables ({overview['cat_count']}):**\n"
                f"`{', '.join(overview['categorical_columns'][:6])}{'...' if overview['cat_count'] > 6 else ''}`\n\n"
                f"**Memory Footprint:** {overview['memory_usage_mb']} MB")

        if "titanic" in dataset_name.lower():
            st.markdown("#### ⚓ About the Titanic Dataset")
            st.caption(
                "The Titanic dataset provides passenger records from the RMS Titanic's maiden voyage in April 1912. "
                "It contains survival records alongside demographic variables (Age, Sex) and socio-economic indicators "
                "(Passenger Class, Fare, Cabin deck)."
            )


# ==============================================================================
# SECTION 2: DATASET EXPLORER
# ==============================================================================
elif menu == "🔍 Dataset Explorer":
    st.markdown('<div class="main-title">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect raw records, inspect dimensions, verify data types, and explore column properties.</div>', unsafe_allow_html=True)

    overview = get_dataset_overview(df_clean)

    # Overview KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Rows", f"{overview['total_rows']:,}")
    k2.metric("Total Columns", overview['total_columns'])
    k3.metric("Total Cells", f"{overview['total_cells']:,}")
    k4.metric("Missing Cells", f"{overview['total_missing']:,}", f"{overview['missing_percentage']}%")
    k5.metric("Duplicate Rows", overview['duplicate_rows'], f"{overview['duplicate_percentage']}%")

    st.markdown("---")

    # Preview controls
    st.markdown("### 📄 **Dataset Preview**")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 4])
    with ctrl_col1:
        n_rows = st.slider("Number of rows to preview:", min_value=5, max_value=50, value=10, step=5)
    with ctrl_col2:
        preview_mode = st.radio("View mode:", ["Head (First rows)", "Tail (Last rows)"], horizontal=True)

    # Column subset selector
    with ctrl_col3:
        selected_preview_cols = st.multiselect(
            "Filter visible columns (optional):",
            options=df_clean.columns.tolist(),
            default=df_clean.columns.tolist()
        )

    preview_df = df_clean[selected_preview_cols] if selected_preview_cols else df_clean
    if preview_mode.startswith("Head"):
        st.dataframe(preview_df.head(n_rows), use_container_width=True)
    else:
        st.dataframe(preview_df.tail(n_rows), use_container_width=True)

    st.markdown("---")

    # Column Details & Types
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🏷️ **Column Catalog & Data Types**")
        missing_summary = get_missing_summary(df_clean)
        st.dataframe(missing_summary, use_container_width=True)

    with col_right:
        st.markdown("### ⚠️ **Dataset Context & Known Limitations**")
        if "titanic" in dataset_name.lower():
            st.markdown("""
            **Domain Background:**
            - **Historical Event:** Sinking of RMS Titanic on April 15, 1912.
            - **Sample Size:** 891 records in standard training split (~40% of the 2,224 passengers & crew).
            - **Target Variable:** `Survived` (0 = Did not survive, 1 = Survived).

            **Key Analytical Limitations:**
            1. **High Missingness in Cabin/Deck:** Over 75% of deck entries are null because records were destroyed or unrecorded for lower classes.
            2. **Age Approximations:** Several infant ages are fractional, and estimated adult ages were inferred post-disaster.
            3. **Survivorship Bias:** Survivors were interviewed more comprehensively than casualties, creating potential documentation asymmetries.
            """)
        else:
            st.markdown(f"""
            **User Dataset Information:**
            - **Source:** User uploaded `{dataset_name}`.
            - **Attributes:** {overview['total_columns']} features detected.

            **Analytical Considerations:**
            - Verify if missing records reflect non-response or true zeros.
            - Inspect whether identifier columns (IDs, keys) should be excluded from statistical analysis.
            """)


# ==============================================================================
# SECTION 3: DATA CLEANING
# ==============================================================================
elif menu == "🧹 Data Cleaning":
    st.markdown('<div class="main-title">Data Cleaning and Preparation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect missing values, duplicates, and outliers. Perform controlled cleaning while preserving the original dataset.</div>', unsafe_allow_html=True)

    # Before and After Summary Cards
    diff = get_cleaning_diff(df_original, df_clean)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Rows", f"{diff['rows_after']:,}", delta=f"{diff['rows_diff']}" if diff['rows_diff'] != 0 else "0")
    c2.metric("Missing Cells", f"{diff['missing_after']:,}", delta=f"{diff['missing_diff']}" if diff['missing_diff'] != 0 else "0", delta_color="inverse")
    c3.metric("Duplicate Rows", f"{diff['duplicates_after']:,}", delta=f"{diff['duplicates_diff']}" if diff['duplicates_diff'] != 0 else "0", delta_color="inverse")
    c4.metric("Memory Usage", f"{diff['memory_after_kb']:.1f} KB")

    st.markdown("---")

    tab_missing, tab_duplicates, tab_outliers, tab_types, tab_audit = st.tabs([
        "🧩 Missing Values",
        "👥 Duplicate Records",
        "🎯 Outlier Detection (IQR)",
        "🔄 Data Type Conversion",
        "📜 Cleaning Audit Log"
    ])

    # --- TAB 1: MISSING VALUES ---
    with tab_missing:
        st.markdown("### 🧩 Missing Value Analysis & Imputation")
        col_fig, col_action = st.columns([3, 2])

        with col_fig:
            fig_miss = plot_missing_values(df_clean)
            st.plotly_chart(fig_miss, use_container_width=True)

        with col_action:
            st.markdown("#### **Handle Missing Values**")
            missing_cols = df_clean.columns[df_clean.isna().sum() > 0].tolist()

            if not missing_cols:
                st.success("🎉 No missing values in any column of the dataset!")
            else:
                col_to_clean = st.selectbox("Select column to impute:", options=missing_cols)
                is_num = pd.api.types.is_numeric_dtype(df_clean[col_to_clean])
                missing_cnt = int(df_clean[col_to_clean].isna().sum())
                st.caption(f"Selected **'{col_to_clean}'** has **{missing_cnt}** missing entries ({is_num and 'Numerical' or 'Categorical'}).")

                if is_num:
                    strategy = st.selectbox(
                        "Imputation Strategy:",
                        options=["median", "mean", "mode", "constant", "drop"],
                        help="Median is recommended for skewed distributions; Mean for symmetric distributions."
                    )
                    custom_num = None
                    if strategy == "constant":
                        custom_num = st.number_input("Enter constant numerical value:", value=0.0)

                    if st.button("Apply Numerical Imputation", use_container_width=True):
                        new_df, info = impute_numerical_column(df_clean, col_to_clean, strategy, custom_num)
                        st.session_state["df_clean"] = new_df
                        st.session_state["cleaning_history"].append(info["description"])
                        st.success(info["description"])
                        st.rerun()
                else:
                    strategy_cat = st.selectbox(
                        "Imputation Strategy:",
                        options=["mode", "constant", "drop"],
                        help="Mode fills with the most frequent category. Constant fills with a placeholder text."
                    )
                    custom_cat = "Missing"
                    if strategy_cat == "constant":
                        custom_cat = st.text_input("Enter category text:", value="Missing")

                    if st.button("Apply Categorical Imputation", use_container_width=True):
                        new_df, info = impute_categorical_column(df_clean, col_to_clean, strategy_cat, custom_cat)
                        st.session_state["df_clean"] = new_df
                        st.session_state["cleaning_history"].append(info["description"])
                        st.success(info["description"])
                        st.rerun()

    # --- TAB 2: DUPLICATES ---
    with tab_duplicates:
        st.markdown("### 👥 Duplicate Record Inspection")
        dup_summary = get_duplicate_summary(df_clean)

        if dup_summary["has_duplicates"]:
            st.warning(f"⚠️ Found **{dup_summary['duplicate_count']}** duplicate row(s) ({dup_summary['duplicate_pct']}% of data).")
            dup_indices = dup_summary["duplicate_indices"]
            st.dataframe(df_clean.loc[dup_indices].head(10), use_container_width=True)

            if st.button("🗑️ Remove All Duplicate Rows", type="primary"):
                deduped_df, dropped = remove_duplicates(df_clean)
                st.session_state["df_clean"] = deduped_df
                msg = f"Removed {dropped} duplicate record(s)."
                st.session_state["cleaning_history"].append(msg)
                st.success(msg)
                st.rerun()
        else:
            st.success("✅ Dataset contains zero duplicate rows.")

    # --- TAB 3: OUTLIERS (IQR) ---
    with tab_outliers:
        st.markdown("### 🎯 Outlier Detection via Interquartile Range (IQR)")
        st.markdown("""
        The **IQR (Tukey's Fences)** method identifies extreme data points without assuming a normal distribution:
        - $IQR = Q_3 - Q_1$
        - $\\text{Lower Fence} = Q_1 - 1.5 \\times IQR$
        - $\\text{Upper Fence} = Q_3 + 1.5 \\times IQR$
        """)

        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.warning("No numerical columns available for outlier detection.")
        else:
            sel_outlier_col = st.selectbox("Select numerical column to inspect:", options=num_cols)
            iqr_mult = st.slider("IQR Multiplier (Fence Factor):", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

            outlier_info = detect_outliers_iqr(df_clean, sel_outlier_col, multiplier=iqr_mult)

            # Metric stats
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Q1 (25%)", outlier_info["q1"])
            m2.metric("Median (50%)", round(df_clean[sel_outlier_col].median(), 2))
            m3.metric("Q3 (75%)", outlier_info["q3"])
            m4.metric("IQR", outlier_info["iqr"])
            m5.metric("Outliers Found", f"{outlier_info['outlier_count']} ({outlier_info['outlier_pct']}%)")

            st.caption(f"Valid Range Fences: **[{outlier_info['lower_bound']} , {outlier_info['upper_bound']}]** | Min: {outlier_info['min_val']} | Max: {outlier_info['max_val']}")

            # Visual boxplot
            fig_box = plot_boxplot(df_clean, sel_outlier_col)
            st.plotly_chart(fig_box, use_container_width=True)

            if outlier_info["outlier_count"] > 0:
                st.markdown("#### **Handle Detected Outliers**")
                action_choice = st.radio(
                    "Select remediation action:",
                    ["Cap values at fences (Winsorization - preserves row count)", "Drop outlier records (removes rows)"],
                    index=0
                )
                action_key = "clip" if "Cap" in action_choice else "drop"

                if st.button("Execute Outlier Handling"):
                    mod_df, audit = handle_outliers_iqr(df_clean, sel_outlier_col, action=action_key, multiplier=iqr_mult)
                    st.session_state["df_clean"] = mod_df
                    st.session_state["cleaning_history"].append(audit["message"])
                    st.success(audit["message"])
                    st.rerun()

    # --- TAB 4: DATA TYPE CONVERSION ---
    with tab_types:
        st.markdown("### 🔄 Column Data Type Inspection & Conversion")
        type_col1, type_col2, type_col3 = st.columns(3)

        with type_col1:
            conv_col = st.selectbox("Select column to cast:", options=df_clean.columns.tolist())
            curr_type = str(df_clean[conv_col].dtype)
            st.caption(f"Current Data Type: `{curr_type}`")

        with type_col2:
            target_type = st.selectbox(
                "Target Data Type:",
                options=["numeric", "integer", "string", "category", "datetime"]
            )

        with type_col3:
            st.write("")
            st.write("")
            if st.button("Convert Data Type", use_container_width=True):
                c_df, success, msg = convert_column_type(df_clean, conv_col, target_type)
                if success:
                    st.session_state["df_clean"] = c_df
                    st.session_state["cleaning_history"].append(msg)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # --- TAB 5: AUDIT TRAIL ---
    with tab_audit:
        st.markdown("### 📜 Session Cleaning Audit Trail")
        if cleaning_history:
            for idx, item in enumerate(cleaning_history, 1):
                st.markdown(f"- **Action {idx}:** {item}")
        else:
            st.info("No cleaning transformations applied yet. Dataset is in original state.")


# ==============================================================================
# SECTION 4: STATISTICAL SUMMARY
# ==============================================================================
elif menu == "📈 Statistical Summary":
    st.markdown('<div class="main-title">Statistical Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous descriptive metrics: central tendency, dispersion, quartiles, asymmetry, and frequency tables.</div>', unsafe_allow_html=True)

    tab_num, tab_cat, tab_detail = st.tabs([
        "🔢 Numerical Statistics",
        "🔤 Categorical Summaries",
        "📊 Detailed Frequency Inspector"
    ])

    with tab_num:
        st.markdown("### 🔢 Descriptive Statistics for Numerical Variables")
        num_summary = get_numerical_summary(df_clean)

        if not num_summary.empty:
            st.dataframe(num_summary, use_container_width=True)

            st.markdown("""
            **Interpretation Guide:**
            - **Mean vs. Median:** A large difference indicates skewness. If Mean > Median, the distribution is right-skewed.
            - **Std Dev & IQR:** Measure variability. The Interquartile Range (IQR = Q3 - Q1) is robust to extreme outliers.
            - **Skewness:** Values outside $[-1, 1]$ indicate substantial asymmetry.
            - **Kurtosis:** Positive kurtosis indicates heavy tails (leptokurtic); negative indicates lighter tails (platykurtic).
            """)
        else:
            st.warning("No numerical features available in this dataset.")

    with tab_cat:
        st.markdown("### 🔤 Summary of Categorical & Text Features")
        cat_summary = get_categorical_summary(df_clean)

        if not cat_summary.empty:
            st.dataframe(cat_summary, use_container_width=True)
            st.caption("Displays unique counts, mode (most common value), and class representation proportions.")
        else:
            st.warning("No categorical features found in this dataset.")

    with tab_detail:
        st.markdown("### 📊 Single-Column Frequency Inspector")
        col_to_inspect = st.selectbox("Select column to inspect frequencies:", options=df_clean.columns.tolist())

        freq_df = get_frequency_distribution(df_clean, col_to_inspect, top_n=12)

        c_table, c_chart = st.columns([1, 1])
        with c_table:
            st.markdown(f"#### **Frequency Table: `{col_to_inspect}`**")
            st.dataframe(freq_df, use_container_width=True)

        with c_chart:
            fig_cat = plot_categorical_distribution(df_clean, col_to_inspect, top_n=10)
            st.plotly_chart(fig_cat, use_container_width=True)


# ==============================================================================
# SECTION 5: VISUAL ANALYSIS
# ==============================================================================
elif menu == "🎨 Visual Analysis":
    st.markdown('<div class="main-title">Visual Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore distributions, outliers, category comparisons, and multivariate interactions with interactive charts.</div>', unsafe_allow_html=True)

    col_types = get_column_types(df_clean)
    num_cols = col_types["numerical"]
    cat_cols = col_types["categorical"]

    vis_type = st.selectbox(
        "Select Visualization Type:",
        [
            "📊 Histograms & Distributions (Numerical)",
            "📦 Boxplots & Outliers (Numerical)",
            "📶 Categorical Frequency Charts (Bar / Count)",
            "🔀 Bivariate Categorical Comparisons (Cross-Tabs)",
            "📍 Scatter Plots & Trendlines (Bivariate Numerical)",
            "🔬 Pair Plot Matrix (Multivariate Seaborn)"
        ]
    )

    st.markdown("---")

    # 1. HISTOGRAMS
    if vis_type.startswith("📊 Histograms"):
        if not num_cols:
            st.warning("No numerical columns available.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                sel_num = st.selectbox("Select numerical column:", options=num_cols)
            with c2:
                bins = st.slider("Number of bins:", min_value=5, max_value=80, value=30, step=5)
            with c3:
                hue_choice = st.selectbox("Group by category (optional):", options=["None"] + cat_cols)
                color_col = None if hue_choice == "None" else hue_choice

            fig = plot_numerical_distribution(df_clean, sel_num, bins=bins, color_col=color_col)
            st.plotly_chart(fig, use_container_width=True)

            # Automated Insight
            mean_v = df_clean[sel_num].mean()
            median_v = df_clean[sel_num].median()
            skew_v = df_clean[sel_num].skew()
            st.info(
                f"💡 **Chart Interpretation for '{sel_num}':** Mean is **{mean_v:.2f}** and Median is **{median_v:.2f}**. "
                f"The distribution has a skewness coefficient of **{skew_v:.2f}** ("
                f"{'right-skewed / long positive tail' if skew_v > 0.5 else ('left-skewed / long negative tail' if skew_v < -0.5 else 'approximately symmetric')})."
            )

    # 2. BOXPLOTS
    elif vis_type.startswith("📦 Boxplots"):
        if not num_cols:
            st.warning("No numerical columns available.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                sel_num = st.selectbox("Select numerical column:", options=num_cols)
            with c2:
                group_choice = st.selectbox("Group by category (optional):", options=["None"] + cat_cols)
                group_col = None if group_choice == "None" else group_choice

            fig = plot_boxplot(df_clean, sel_num, group_by=group_col)
            st.plotly_chart(fig, use_container_width=True)

            outlier_info = detect_outliers_iqr(df_clean, sel_num)
            st.info(
                f"💡 **Chart Interpretation:** Box represents the middle 50% of records (Q1 to Q3). "
                f"The whiskers extend to 1.5x IQR boundaries. Values beyond whiskers (**{outlier_info['outlier_count']} points**, {outlier_info['outlier_pct']}%) "
                "are flagged as potential statistical outliers."
            )

    # 3. CATEGORICAL FREQUENCY CHARTS
    elif vis_type.startswith("📶 Categorical"):
        if not cat_cols:
            st.warning("No categorical columns available.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                sel_cat = st.selectbox("Select categorical column:", options=cat_cols)
            with c2:
                top_k = st.slider("Max categories to show:", min_value=3, max_value=25, value=10)

            fig = plot_categorical_distribution(df_clean, sel_cat, top_n=top_k)
            st.plotly_chart(fig, use_container_width=True)

            top_mode = df_clean[sel_cat].mode().dropna()
            mode_label = top_mode.iloc[0] if not top_mode.empty else "N/A"
            st.info(f"💡 **Chart Interpretation:** Most frequent category is **'{mode_label}'**. Inspect whether any class suffers from heavy under-representation.")

    # 4. BIVARIATE CATEGORICAL
    elif vis_type.startswith("🔀 Bivariate"):
        if len(cat_cols) < 2 and ("Survived" not in df_clean.columns):
            st.warning("Need at least 2 categorical or low-cardinality columns.")
        else:
            all_cat_like = cat_cols + [c for c in num_cols if df_clean[c].nunique() <= 10 and c not in cat_cols]
            c1, c2, c3 = st.columns(3)
            with c1:
                col_feat = st.selectbox("Select feature category (X-axis):", options=all_cat_like, index=0)
            with c2:
                default_target_idx = 1 if len(all_cat_like) > 1 else 0
                if "Survived" in all_cat_like:
                    default_target_idx = all_cat_like.index("Survived")
                col_targ = st.selectbox("Select target category (Grouping):", options=all_cat_like, index=default_target_idx)
            with c3:
                as_pct = st.checkbox("Show as percentage within category (%)", value=True)

            fig = plot_bivariate_categorical(df_clean, col_feat, col_targ, as_percentage=as_pct)
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"💡 **Chart Interpretation:** Compares proportional splits of **'{col_targ}'** across different levels of **'{col_feat}'**.")

    # 5. SCATTER PLOTS
    elif vis_type.startswith("📍 Scatter"):
        if len(num_cols) < 2:
            st.warning("Need at least 2 numerical columns for scatter plots.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x_col = st.selectbox("X-axis (Feature):", options=num_cols, index=0)
            with c2:
                y_col = st.selectbox("Y-axis (Feature):", options=num_cols, index=min(1, len(num_cols) - 1))
            with c3:
                color_opt = st.selectbox("Color hue (optional):", options=["None"] + cat_cols + num_cols)
                hue_c = None if color_opt == "None" else color_opt
            with c4:
                add_trend = st.checkbox("Add OLS Trendline", value=True)

            fig = plot_scatter(df_clean, x_col, y_col, color_col=hue_c, add_trendline=add_trend)
            st.plotly_chart(fig, use_container_width=True)

            # Compute correlation
            valid_sub = df_clean[[x_col, y_col]].dropna()
            if len(valid_sub) > 2:
                r_val = valid_sub[x_col].corr(valid_sub[y_col])
                st.info(f"💡 **Correlation:** Pearson r between **'{x_col}'** and **'{y_col}'** is **{r_val:.3f}** (R² = {r_val**2:.3f}).")

    # 6. PAIR PLOT
    elif vis_type.startswith("🔬 Pair Plot"):
        if len(num_cols) < 2:
            st.warning("Need at least 2 numerical columns for a pair plot.")
        else:
            st.markdown("Generates a comprehensive grid of bivariate scatter plots and univariate KDE distributions.")
            c1, c2 = st.columns([3, 1])
            with c1:
                selected_pair_cols = st.multiselect(
                    "Select numerical columns (2 to 5 recommended for performance):",
                    options=num_cols,
                    default=num_cols[:min(4, len(num_cols))]
                )
            with c2:
                pair_hue = st.selectbox("Hue grouping column (optional):", options=["None"] + cat_cols)
                hue_val = None if pair_hue == "None" else pair_hue

            if len(selected_pair_cols) >= 2:
                with st.spinner("Rendering pairwise Seaborn plot..."):
                    fig_pair = plot_pairplot(df_clean, selected_pair_cols, hue_col=hue_val)
                    st.pyplot(fig_pair)
            else:
                st.warning("Please select at least 2 numerical columns.")


# ==============================================================================
# SECTION 6: CORRELATION ANALYSIS
# ==============================================================================
elif menu == "🔗 Correlation Analysis":
    st.markdown('<div class="main-title">Correlation and Key Influencing Factors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluate numerical correlations, identify top association pairs, and explore target relationships.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Crucial Scientific Distinction: Correlation ≠ Causation</strong><br>
        A statistical correlation measures how variables co-vary, but does <em>not</em> prove that one variable causes changes in the other.
        Observed patterns may be driven by unmeasured confounding factors, reverse causality, or coincidence.
        Always treat discovered patterns as exploratory hypotheses requiring domain validation.
    </div>
    """, unsafe_allow_html=True)

    corr_method = st.radio(
        "Select Correlation Metric:",
        ["pearson", "spearman"],
        format_func=lambda x: "Pearson (Measures linear relationships)" if x == "pearson" else "Spearman Rank (Measures monotonic rank relationships)",
        horizontal=True
    )

    corr_matrix = compute_correlation_matrix(df_clean, method=corr_method)

    if corr_matrix.empty:
        st.warning("Insufficient numerical columns to compute correlations.")
    else:
        # Heatmap
        st.markdown(f"### 🌡️ **{corr_method.capitalize()} Correlation Heatmap**")
        fig_heat = plot_correlation_heatmap(corr_matrix, title=f"{corr_method.capitalize()} Correlation Matrix")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # Top Positive and Negative Correlations
        st.markdown("### 🔝 **Strongest Correlation Pairs**")
        strong_pairs = get_strongest_correlations(corr_matrix, top_n=5)
        pos_col, neg_col = st.columns(2)

        with pos_col:
            st.markdown("#### **Top Positive Correlations (Direct Relationship)**")
            if not strong_pairs["positive"].empty:
                st.dataframe(strong_pairs["positive"], use_container_width=True)
            else:
                st.caption("No positive correlations found.")

        with neg_col:
            st.markdown("#### **Top Negative Correlations (Inverse Relationship)**")
            if not strong_pairs["negative"].empty:
                st.dataframe(strong_pairs["negative"], use_container_width=True)
            else:
                st.caption("No negative correlations found.")

        st.markdown("---")

        # Target Column Deep-Dive
        st.markdown("### 🎯 **Target Variable Influencing Factor Explorer**")
        st.caption("Select a primary outcome or target column to evaluate how every other variable interacts with it.")

        all_cols = df_clean.columns.tolist()
        default_target = "Survived" if "Survived" in all_cols else all_cols[0]
        selected_target = st.selectbox("Select Target / Outcome Column:", options=all_cols, index=all_cols.index(default_target))

        other_features = [c for c in all_cols if c != selected_target]
        selected_feature = st.selectbox("Select Feature to Test Against Target:", options=other_features)

        if st.button("Analyze Relationship with Target", type="primary"):
            res = analyze_target_relationship(df_clean, selected_target, selected_feature)

            st.markdown(f"#### **Statistical Findings: `{selected_feature}` vs `{selected_target}`**")
            st.info(res.get("interpretation", "Analysis complete."))

            if res.get("type") == "num_vs_num":
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pearson r", res["pearson_r"])
                m2.metric("Spearman rho", res["spearman_rho"])
                m3.metric("R-Squared (R²)", res["r_squared"])
                m4.metric("Regression Slope", res["slope"])
                fig_sc = plot_scatter(df_clean, selected_feature, selected_target, add_trendline=True)
                st.plotly_chart(fig_sc, use_container_width=True)

            elif res.get("type") == "cat_target_vs_num_feature":
                st.dataframe(res["group_table"], use_container_width=True)
                fig_bx = plot_boxplot(df_clean, selected_feature, group_by=selected_target)
                st.plotly_chart(fig_bx, use_container_width=True)

            elif res.get("type") == "cat_vs_cat":
                c_tbl, c_pct = st.columns(2)
                with c_tbl:
                    st.markdown("**Contingency Counts:**")
                    st.dataframe(res["contingency_table"], use_container_width=True)
                with c_pct:
                    st.markdown("**Category Proportions (%):**")
                    st.dataframe(res["contingency_pct_table"], use_container_width=True)

                if res.get("chi2") is not None:
                    ch1, ch2, ch3 = st.columns(3)
                    ch1.metric("Chi-Square Statistic (χ²)", res["chi2"])
                    ch2.metric("p-value", res["p_value"])
                    ch3.metric("Cramer's V (Effect Size)", res["cramers_v"])

                fig_bi = plot_bivariate_categorical(df_clean, selected_feature, selected_target)
                st.plotly_chart(fig_bi, use_container_width=True)


# ==============================================================================
# SECTION 7: INSIGHTS AND REPORT
# ==============================================================================
elif menu == "📝 Insights and Report":
    st.markdown('<div class="main-title">Insights and Structured EDA Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review data-driven findings and export the complete 13-heading academic EDA report.</div>', unsafe_allow_html=True)

    # Key findings cards
    target_candidate = "Survived" if "Survived" in df_clean.columns else None
    findings = generate_key_findings(df_clean, target_col=target_candidate)

    st.markdown("### 🔍 **Computed Key Findings (Preliminary Observations)**")
    st.caption("These findings are dynamically computed from your active dataset records.")

    for f in findings:
        st.markdown(f"""
        <div class="notice-box">
            <strong>[{f['category']}] {f['title']}</strong><br>
            {f['description']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Generate full 13-heading report
    report_md = generate_eda_markdown_report(
        df_raw=df_original,
        df_cleaned=df_clean,
        cleaning_history=cleaning_history,
        dataset_name=dataset_name,
        target_col=target_candidate
    )

    # Downloads Row
    st.markdown("### 📥 **Download Deliverables**")
    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="📄 Download EDA Report (.md)",
            data=report_md,
            file_name=f"EDA_Report_{dataset_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with d2:
        csv_data = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Cleaned Dataset (.csv)",
            data=csv_data,
            file_name=f"Cleaned_{dataset_name.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # Live Markdown Report Preview
    st.markdown("### 📖 **Live Report Preview (13 Required Academic Headings)**")
    with st.expander("Click to view complete formatted report inline", expanded=True):
        st.markdown(report_md)
