# Employee Performance & Productivity — Data Mining

Predicting **employee attrition, performance, and satisfaction** from workforce data using exploratory analysis and machine learning. Built as the capstone project for **ALY 6040: Data Mining Applications** (Northeastern University, Fall 2024).

## Business Problem

Organizations struggle to identify which factors most influence employee outcomes. Using a dataset of **100,000 employee records across 20 variables**, this project investigates three questions:

- **Performance:** Which factors (education level, work hours, project involvement) correlate most with high performance scores?
- **Retention:** Can we predict resignation risk from overtime, satisfaction, and promotion history?
- **Satisfaction:** How do remote-work frequency, team size, and training hours affect employee satisfaction and tenure?

## Dataset

[Employee Performance and Productivity Data](https://www.kaggle.com/) (Kaggle) — 100,000 rows × 20 columns, no missing values.

| Category | Features |
|----------|----------|
| **Demographics** | Employee_ID, Department, Gender, Age, Job_Title, Education_Level |
| **Tenure & Role** | Hire_Date, Years_At_Company, Team_Size, Promotions |
| **Work patterns** | Work_Hours_Per_Week, Projects_Handled, Overtime_Hours, Remote_Work_Frequency, Training_Hours, Sick_Days |
| **Compensation** | Monthly_Salary |
| **Outcomes** | Performance_Score, Employee_Satisfaction_Score, **Resigned** (target) |

## Approach

1. **Data selection & proposal** — framed the business problem and validated dataset suitability.
2. **Exploratory Data Analysis (EDA)** — distributions, correlations, and feature relationships across performance, satisfaction, and attrition.
3. **Modeling** — `RandomForestClassifier` within a scikit-learn `Pipeline` (`StandardScaler` + `OneHotEncoder` via `ColumnTransformer`), tuned with `GridSearchCV` and evaluated with classification report and ROC-AUC.
4. **Final report** — synthesized findings into actionable HR recommendations for retention and performance optimization.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` / `Seaborn` · `Jupyter`

## Repository Structure

```
employee-performance-data-mining/
├── data/                          # Raw dataset (100K records)
├── notebooks/
│   └── 01_dataset_proposal_and_eda.ipynb
├── analysis/
│   ├── modeling.md / modeling.html   # Modeling walkthrough + outputs
│   └── images/                       # Generated charts
└── reports/                       # Module write-ups (PDF)
    ├── 01_dataset_proposal.pdf
    ├── 02_eda.pdf
    ├── 02_milestone1.pdf
    ├── 05_modeling.pdf
    └── 06_final_report.pdf
```

## Author

**Shraddha Gupte** — MS Data Analytics, Northeastern University · MBA (HR)
People Analytics professional | [LinkedIn](https://linkedin.com/in/shraddhapg) · [Portfolio](https://gupte-shra.github.io)

> Academic project completed for ALY 6040 (Prof. Arasu Narayan). Dataset sourced from Kaggle for educational use.
