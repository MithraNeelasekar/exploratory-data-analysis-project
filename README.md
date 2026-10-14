# 📊 Exploratory Data Analysis (EDA) Project

An end-to-end, interactive, and beginner-friendly Exploratory Data Analysis (EDA) application built with **Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, and Streamlit**. 

Developed for a college data analytics assignment, this project provides a systematic framework to explore datasets, clean dirty data, compute rigorous descriptive statistics, detect outliers using the Interquartile Range (IQR) method, examine correlation drivers, and generate academic-grade 13-heading EDA reports.

---

## 📌 Repository Information

- **Repository Name:** `exploratory-data-analysis-project`
- **Short Description:** *An interactive Python & Streamlit web dashboard for Exploratory Data Analysis (EDA), featuring automated cleaning, IQR outlier detection, interactive Plotly visualizations, correlation modeling, and dynamic 13-section report generation.*
- **Target Audience:** College faculty evaluation, viva presentations, data analytics students, and beginner Python developers.

---

## 🎯 1. Project Objective

The primary objective of this project is to analyze a structured dataset to uncover fundamental patterns, trends, and variable interactions through data-driven exploration.

### Key Competencies Demonstrated:
1. **Statistical Summaries:** Five-number summaries, mean, median, mode, standard deviation, variance, skewness, and kurtosis.
2. **Data Cleaning & Hygiene:** Non-destructive missing value imputation, duplicate removal, and Tukey's 1.5x IQR outlier winsorization.
3. **Interactive Visualizations:** Histograms with distribution curves, boxplots, count charts, OLS regression scatter plots, and annotated correlation heatmaps.
4. **Correlation & Target Drivers:** Pearson linear correlation, Spearman rank monotonic correlation, ANOVA group tests, and Chi-Square contingency analysis.
5. **Academic Reporting:** Dynamic compilation of a 13-section EDA report rendered in Markdown and downloadable alongside the cleaned dataset.

---

## 🛠️ 2. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.14 | Core programming language |
| **Data Manipulation** | Pandas, NumPy | High-performance tabular data structures and numerical operations |
| **Statistical Modeling** | SciPy | ANOVA, Chi-Square contingency, Fisher-Pearson skewness, OLS regressions |
| **Interactive UI** | Streamlit | Modern reactive web dashboard with widgets, tabs, and sidebar navigation |
| **Interactive Visuals** | Plotly Express & Graph Objects | Dynamic charts with hover tooltips, zoom, and responsive layouts |
| **Static Visuals** | Matplotlib, Seaborn | Aesthetic multi-variable pair plot matrices |
| **Testing** | Unittest | 29 automated test cases covering computations, cleaning, and edge cases |

---

## 📁 3. Project Structure

```text
exploratory-data-analysis-project/
│
├── app.py                 # Streamlit web dashboard application
├── eda_analysis.py        # Descriptive statistics, distributions, correlation & target analysis
├── data_cleaning.py       # Missing imputation, duplicate handling, IQR outlier winsorization
├── visualizations.py      # Plotly & Seaborn figure generation functions
├── report_generator.py    # Dynamic 13-heading academic EDA report generator
├── test_eda.py            # Automated unit test suite (29 tests)
│
├── data/
│   └── titanic.csv        # Bundled offline Titanic dataset (891 rows × 16 columns)
│
├── requirements.txt       # Python project dependencies
├── .gitignore             # Git exclusions (caches, virtual environments, artifacts)
└── README.md              # Project documentation and presentation guide
```

---

## 🚢 4. Dataset Overview

### Default Dataset: Titanic Passenger Records
The application includes a bundled local copy of the **Titanic dataset** (`data/titanic.csv`), requiring **no internet connection** to run.
- **Dimensions:** 891 observations (rows) × 16 features (columns).
- **Attributes:**
  - `PassengerId` (int): Unique identifier.
  - `Survived` (int): Target variable (0 = Did not survive, 1 = Survived).
  - `Pclass` / `Class` (int/str): Ticket class (1st = Upper, 2nd = Middle, 3rd = Lower).
  - `Sex` (str): Gender (male / female).
  - `Age` (float): Passenger age in years (contains ~19.9% missing values).
  - `SibSp` (int): Count of siblings / spouses aboard.
  - `Parch` (int): Count of parents / children aboard.
  - `Fare` (float): Passenger fare amount in British pounds.
  - `Embarked` / `EmbarkTown` (str): Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
  - `Deck` (str): Cabin deck letter (contains ~77% missing values).
  - `Alone` (bool): Indicator if passenger traveled without family.

### Custom CSV Upload
Users can upload any arbitrary CSV dataset via the sidebar. The dashboard automatically inspects dimensions, segregates numerical from categorical features, audits missingness, and adapts all visualizers and statistical tables dynamically.

---

## 🚀 5. Installation & Local Execution

Follow these straightforward steps to run the application on your computer:

### Step 1: Open Your Terminal / PowerShell
Navigate to the project root directory:
```powershell
cd "c:\Users\MITHRA N\Desktop\Thiranex 3"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Automated Test Suite
Ensure all 29 unit tests pass:
```powershell
python -m unittest test_eda.py -v
```

### Step 5: Launch the Interactive Streamlit Dashboard
Run the Streamlit application using `python -m streamlit`:
```powershell
python -m streamlit run app.py
```

The application will launch automatically in your default web browser at:
👉 **`http://localhost:8501`**

---

## 🖥️ 6. Dashboard Sections Walkthrough

1. **🏠 Home:**
   - Highlights project objectives, data definitions, and high-level KPI cards (Total Records, Features, Missing %, Duplicate Count).
   - Explains John Tukey's EDA workflow and how to navigate the system.
2. **🔍 Dataset Explorer:**
   - Preview dataset rows (Head / Tail views with customizable row count).
   - Filter visible column subsets.
   - Column catalog displaying inferred data types, non-null counts, and missing percentages.
   - Dataset domain context and historical limitations.
3. **🧹 Data Cleaning:**
   - **Missing Values:** Interactive imputation (Mean, Median, Mode, Constant, Drop Rows).
   - **Duplicate Records:** Duplicate detection and removal with instant row recalculation.
   - **Outlier Detection:** Tukey's 1.5x IQR method calculation, outlier counts, and optional Winsorization (capping at fences) or dropping.
   - **Type Conversion:** Safe column casting to numeric, integer, category, or datetime.
   - **Audit Log & Diff:** Live comparative diff (Rows, Missing cells, Duplicates before vs after). Original dataset is never modified.
4. **📈 Statistical Summary:**
   - Numerical table: Mean, Std Dev, Variance, Min, Q1, Median, Q3, Max, Range, IQR, Mode, Skewness, Kurtosis.
   - Categorical summary: Unique count, Mode, Frequency, Mode %.
   - Single-column frequency inspector with interactive percentages and bar charts.
5. **🎨 Visual Analysis:**
   - **Histograms:** Distribution shapes, bin controls, optional hue grouping, mean and median lines.
   - **Boxplots:** Outlier points identification and cross-category groupings.
   - **Bar Charts:** Categorical counts and proportions.
   - **Bivariate Cross-Tabs:** Subgroup percentage and count comparisons.
   - **Scatter Plots:** Bivariate numerical scatter with OLS regression lines and Pearson $r$.
   - **Pair Plots:** Multivariate Seaborn matrix for pairwise distributions.
6. **🔗 Correlation Analysis:**
   - Pearson (linear) and Spearman (rank) correlation heatmaps with cell annotations.
   - Tables of top positive and top negative correlation pairs.
   - **Target Variable Deep Dive:** Select any target (e.g. `Survived`) to compute ANOVA F-tests, Chi-Square ($\chi^2$), Cramer's V, or linear regressions.
   - Educational banner emphasizing that **correlation does not imply causation**.
7. **📝 Insights and Report:**
   - Automated, factually verified findings computed directly from the dataset.
   - Full 13-heading academic EDA report rendered live in Markdown.
   - One-click buttons to download the report as `.md` and the cleaned dataset as `.csv`.

---

## 📄 7. The 13-Heading Structured EDA Report

The report generator strictly conforms to the 13 required academic sections:

1. **Project Title**
2. **Objective**
3. **Dataset Description**
4. **Data Overview**
5. **Data Quality and Missing Values**
6. **Data Cleaning Performed**
7. **Statistical Summary**
8. **Important Visualizations**
9. **Correlation Analysis**
10. **Key Findings**
11. **Limitations**
12. **Conclusion**
13. **Future Improvements**

---

## 🎓 8. How to Demonstrate to Faculty (Viva & Evaluation Guide)

When presenting this project to your college professor or external evaluator, follow this 5-minute demonstration script:

### Recommended Presentation Flow:
1. **Introduction (1 min):**
   - Start on the **Home** tab. State the objective: *"This project performs an exploratory data analysis to uncover statistical distributions, audit data hygiene, test correlations, and identify key influencing factors without assuming causal links."*
   - Point out the 4 KPI cards showing instant dataset metrics.
2. **Data Cleaning & IQR Outliers (1.5 min):**
   - Switch to **Data Cleaning**.
   - Show the **Missing Values** tab: Explain why `Age` is imputed with the **Median** rather than the Mean (because Age exhibits skewness and outliers, making the median a more robust metric).
   - Show the **Outlier Detection (IQR)** tab: Select `Fare`. Explain how Q1, Q3, and the 1.5x IQR fences are computed ($Q_3 + 1.5 \times IQR$). Show how Winsorization caps extreme values without dropping rows.
   - Point out the **Audit Log** and the **Before & After Diff** metric cards. Emphasize that the original data is preserved in memory.
3. **Statistical Analysis & Visualizations (1 min):**
   - Switch to **Visual Analysis**. Show the `Fare` histogram with the red dashed mean line and green dotted median line.
   - Demonstrate the **Bivariate Categorical** chart: `Sex` vs `Survived` (showing female survival was ~74% compared to male survival at ~19%, reflecting the maritime "women and children first" protocol).
4. **Correlation vs Causation (1 min):**
   - Switch to **Correlation Analysis**.
   - Show the Pearson heatmap.
   - Explain the strong negative correlation between `Fare` and `Pclass` ($r \approx -0.55$, because 1st class tickets cost more than 3rd class).
   - Highlight the **Warning Callout:** Explain to faculty that correlation indicates co-variation, not causation, and mention potential confounding variables.
5. **Report & Deliverables (30 sec):**
   - Navigate to **Insights and Report**.
   - Show the generated preliminary findings and click **Download EDA Report (.md)** and **Download Cleaned Dataset (.csv)**.

### Common Viva Questions & Model Answers:
- **Q: Why use Median instead of Mean for missing numerical data?**
  - *Answer:* The mean is pulled by extreme outliers (skewness). The median represents the 50th percentile and is resistant to outliers.
- **Q: How does the IQR method detect outliers?**
  - *Answer:* It computes the spread of the middle 50% ($IQR = Q_3 - Q_1$). Any point less than $Q_1 - 1.5 \times IQR$ or greater than $Q_3 + 1.5 \times IQR$ is classified as a Tukey fence outlier.
- **Q: Why can't we claim `Pclass` caused survival based only on correlation?**
  - *Answer:* Correlation does not prove causality. Higher-class passengers were located on upper decks closer to lifeboats (spatial proximity confounder) and received evacuation priority, which are the physical mechanisms rather than the ticket price itself.

---

## 📤 9. Steps to Upload the Project to GitHub

To push this project to your GitHub account:

### Step 1: Initialize Git in the Project Directory
```powershell
git init
```

### Step 2: Configure Your Git Identity (if not already set)
```powershell
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 3: Stage All Project Files
```powershell
git add .
```
*(The provided `.gitignore` automatically prevents temporary caches, virtual environments, and artifacts from being added.)*

### Step 4: Create the Initial Commit
```powershell
git commit -m "feat: complete exploratory data analysis project with streamlit dashboard and unit tests"
```

### Step 5: Link Your Remote GitHub Repository
1. Go to [GitHub.com](https://github.com) and click **New Repository**.
2. Name the repository: `exploratory-data-analysis-project`.
3. Set visibility to **Public** (or Private). Do *not* initialize with a README, license, or gitignore since we already have them.
4. Copy the repository URL and execute:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/exploratory-data-analysis-project.git
git branch -M main
git push -u origin main
```

---

## 🔮 10. Suggested Future Improvements

1. **Multivariate Imputation:** Implement iterative model-based imputation such as MICE (Multivariate Imputation by Chained Equations) or KNN Imputer.
2. **Automated Feature Engineering:** Generate composite interaction terms (e.g., `FamilySize = SibSp + Parch + 1`, Title extraction from passenger names).
3. **Supervised Predictive Modeling:** Add an interactive Machine Learning playground tab with Scikit-Learn (Logistic Regression, Random Forest, ROC-AUC curves).
4. **PDF Report Export:** Integrate WeasyPrint or ReportLab to export formatted PDF reports with embedded charts alongside Markdown.
5. **Automated Hypothesis Testing:** Add one-click Wilcoxon-Mann-Whitney and Kruskal-Wallis non-parametric significance testing for skewed continuous groups.

---

## 📜 License & Acknowledgments
- Developed for academic exploratory data analysis coursework.
- Built using open-source Python libraries (Streamlit, Pandas, Plotly, Seaborn, SciPy).
- Default dataset provided via standard maritime historical records (RMS Titanic passenger manifest).
