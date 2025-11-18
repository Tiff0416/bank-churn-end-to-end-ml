# Bank Churn Prediction & Customer Segmentation

End-to-end Machine Learning + SQL Pipeline + SHAP Explainability 

This project analyzes credit card customer churn using the BankChurners dataset.
It covers SQL-based data cleaning, exploratory analysis, K-Means segmentation, churn modeling (Logistic Regression, Random Forest, XGBoost), and customer-level explainability via SHAP.
The goal is to produce actionable business recommendations supported by data.

---

# Project Structure
```
BankChurners-Project/
│
├── churn_dashboard/
│ └── app.py # Streamlit ROI simulator
│
├── data/
│ ├── raw/ # Original dataset (not uploaded to GitHub)
│ │ └── BankChurners.csv
│ │
│ └── processed/ # Cleaned + analysis-generated artifacts
│ ├── cleaned_data.csv # Cleaned dataset from SQL/Python pipeline
│ ├── bank_churners_clean.csv # Alternative cleaned version (SQL export)
│ ├── cluster_summary.csv # Summary stats for each KMeans cluster
│ ├── cluster_table.csv # Full cluster assignment table
│ ├── shap_summary.csv # SHAP feature importance summary
│ ├── roi_simulation_data.csv # ROI scenario curve used in dashboard
│ ├── X_ready.csv # Model-ready feature matrix
│ └── y_ready.csv # Model-ready target vector
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_modeling.ipynb
│ └── 04_segmentation.ipynb
│
├── sql/
│ ├── 01_import_data.sql
│ └── etl_bankchurners.sql
│
├── src/
│ └── functions.py # Helper functions for modeling & plotting
│
├── report/
│ ├── figures/
│ └── presentation/
│
├── FinalProject_Presentation.pdf
└── README.md
```
---

# Data Cleaning & ETL (SQL Server)

- Built a structured ETL pipeline using SQL Server with separate **staging** and **ODS** layers.
- Loaded raw CSV into `stg.BankChurners_raw` using `BULK INSERT`.
- Standardized fields by trimming whitespace, converting empty strings to NULL, and casting data into proper types.
- Created a clean analytical dataset in `ods.BankChurners` for downstream EDA and modeling.


---
# Exploratory Data Analysis (EDA) – Key Takeaways

- **46-feature dataset with imbalanced churn**  
  Converted `Attrition_Flag` into a binary target. Churn rate is **~16%**, confirming a **heavily imbalanced** classification problem.

- **Churn is behavior-driven, not demographic-driven**  
  Among 46 features, age, gender, income, and marital status show **near-zero correlation** with churn. Behavioral variables dominate.

- **Strong behavioral churn signals**  
  Churned customers consistently show:  
  - **Lower transaction counts & amounts**  
  - **More inactive months**  
  - **More customer-service contacts**  
  - **Fewer product relationships**  
  - **Declining Q4→Q1 spend/activity trends**  
  These patterns indicate **early disengagement before attrition**.

- **Insights guided feature engineering**  
  Engagement intensity, inactivity, utilization, and trend-based variables were prioritized for modeling and retention-strategy design.

---
# Feature Engineering

To prepare the dataset for churn prediction, I transformed the original 46 features into a clean and fully model-ready dataset focused on customer behavior and engagement.

Key steps included:

- **Created meaningful customer segments**  
  Added features such as tenure groups, age groups, and dependent indicators to better capture lifecycle differences among customers.

- **Strengthened behavioral signals**  
  Based on EDA findings, I engineered features that highlight changes in spending, transaction frequency, inactivity, and credit usage—factors closely linked to churn risk.

- **Improved numerical stability**  
  Applied log transformations to highly skewed financial variables (e.g., spending, credit limit, open-to-buy) to help models learn more effectively.

- **Prepared all features for machine learning**  
  One-hot encoded categorical variables (while keeping useful categories like “Unknown”), removed ID-only fields, and ensured the final dataset contained only numeric, clean inputs with no missing values.

The result is a high-quality modeling dataset (`X_ready.csv` and `y_ready.csv`) that captures the key behavioral patterns behind customer churn and supports strong predictive performance.

---
# Churn Prediction Modeling

To predict customer churn, I trained three machine-learning models:

- Logistic Regression  
- Random Forest  
- XGBoost (with imbalance handling)

Because churners make up only ~16% of the dataset, the models were trained with class-imbalance techniques to improve sensitivity to at-risk customers.

---

## Model Performance (Test Set)

| Model               | Accuracy | Recall | F1   | ROC-AUC |
| ------------------- | -------- | ------ | ---- | ------- |
| Logistic Regression | 0.869    | 0.846  | 0.674| 0.938   |
| Random Forest       | 0.956    | 0.825  | 0.858| 0.987   |
| **XGBoost**         | **0.968**| **0.926** | **0.903** | **0.993** |

**XGBoost performed best**, especially in recall (ability to identify churners) and overall predictive power (ROC-AUC).

---

## Explainability (SHAP)

SHAP explainability was used to understand why customers churn.  
The strongest drivers of churn are all **behavioral**:

- Low transaction frequency and spending  
- Declining recent activity (Q4→Q1 drop)  
- High inactivity  
- Higher customer-service contact frequency  
- Fewer product relationships

In contrast, demographic attributes contribute very little.

These insights directly support the later segmentation and retention-strategy design.

---
# Customer Segmentation & Targeting

After building the churn model, I used **K-Means clustering** to segment customers based on the same behavioral drivers that SHAP identified as most important:

- Transaction frequency and amount  
- Revolving balance and credit limit  
- Recent activity trends (Q4 vs Q1)  
- Inactivity and customer-service contacts  
- Product relationship depth  

This produced **4 distinct behavioral segments**:

| Cluster | Size   | Churn | Description                                  |
| ------- | ------ | ----- | -------------------------------------------- |
| 0       | 1,234  | 16%   | Mid-activity, mid-value, early disengagers   |
| 1       | 4,577  | 4%    | High-activity, very loyal customers          |
| 2       | 3,174  | 38%   | Very low activity, highest churn, almost lost|
| 3       | 1,142  | 4%    | Very high spenders with high credit limits   |

Instead of targeting everyone, I focused on **Cluster 0**:

- They still have **meaningful spend and credit limits**, so retention has financial impact.  
- Their patterns show **early signs of disengagement** (more inactive months, declining usage), which is still reversible.  
- Their churn rate (16%) leaves room for improvement without the very low engagement seen in Cluster 2.

---

# Reactivation Strategy, ROI, and Streamlit App

For Cluster 0, I designed a simple, testable reactivation campaign:

- **Offer**: e.g., \$10 cashback if the customer makes **3 transactions within 30 days**.  
- **Test design**: A/B test inside Cluster 0 (Treatment vs Control).  
- **KPIs**: churn rate, transaction count, and revenue per user over 60 days.

I then simulated different cashback levels (e.g., \$5, \$10, \$15…) to estimate:

- Expected churn reduction  
- How many extra customers would be retained  
- Incremental revenue vs. campaign cost  
- Net profit and ROI

These simulation outputs are connected to a **Streamlit dashboard**, where business users can:

- Select a target segment (e.g., Cluster 0)  
- Adjust the cashback/offer level using a slider  
- See the projected **ROI vs. retention lift** update in real time

This ties the project together end-to-end:

> **Model → Segmentation → Target Selection → Campaign Design → ROI Simulation → Streamlit App for business decisions.**

---
# Experimental design (proposed A/B test)
I designed a simple A/B testing framework (conceptual) for Cluster 0:

- Split the segment into Treatment vs Control.
- Offer cashback to the treatment group and observe changes in churn, activity, and revenue.
- Use the Streamlit app to simulate different offer levels and see the expected churn reduction and ROI.

The current app focuses on **simulation and decision support**, showing how different campaign designs would perform before running a real experiment.

---
# Business Recommendations

Based on behavioral signals, churn drivers, and ROI analysis, I recommend:

1. Prioritize Cluster 0 for early-intervention retention programs.
    They offer the highest return per dollar spent and are still recoverable.

2. Launch a low-friction engagement incentive (e.g., $10 activity booster).
    Designed to directly target the strongest churn driver—declining transaction activity.

3. Validate impact through a controlled experiment (A/B test).
    Measure churn reduction, incremental revenue, and cost effectiveness.

4. Use the Streamlit dashboard to continuously refine incentive levels.
    Business users can evaluate campaign profitability under different retention lift assumptions.

This recommendation aims to maximize financial impact, minimize wasteful spend on low-potential segments, and create a scalable framework for future churn-prevention initiatives.

---
# ROI Interpretation (Directly Reflecting the Dashboard)

## Live Demo (Streamlit App)

Interactive ROI Simulator & Churn Dashboard:

https://bank-churn-end-to-end-ml-5pqiiyrcbuujqg97zl6wi7.streamlit.app/

Streamlit simulator shows:

With Retention Lift = 4.5%,
→ Incremental customers = 56
→ Net Profit = $4,319
→ ROI = 0.35x

This matches the expected lift generated from the recommended $10 interventions, and visually aligns with your ROI vs. Retention Lift Curve plot.

---
# How to Run the Project
Clone the repository

- Execute SQL scripts in sql/ to generate processed dataset

- Run notebooks in order:

    1. 01_data_exploration.ipynb

    2. 02_feature_engineering.ipynb

    3. 03_modeling.ipynb

    4. 04_segamentation.ipynb

(Optional) Open dashboard for ROI visualization.

---
# Tech Stack
- Python (Pandas, NumPy, Scikit-Learn, XGBoost, SHAP)

- SQL Server (ETL, ODS Pipeline)

- Tableau

- Matplotlib / Seaborn

- Git & GitHub

---
# License
MIT License
