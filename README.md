# Churn prediction


## 1. 📌 Business Understanding (provided by Lead)

Business Goal:
Reduce customer churn and increase customer lifetime value.

Data Science Objective:
Build a predictive model to identify customers likely to churn in the next month.

Success Metrics:

F1-Score above 0.80 on test data.

Actionable insights into churn drivers.

Business interpretability: Why are customers churning?

## 2. 📦 Data Acquisition

Dataset:
Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Telco-Customer-Churn.csv

## 3. 🧹 Data Cleaning & Preprocessing

Tasks:

Remove duplicate rows (if any)

Handle missing values (TotalCharges has blank values that need conversion)

Convert categorical variables to numerical:

One-hot encode gender, InternetService, Contract, etc.

Label encode Churn → 1 for "Yes", 0 for "No"

Scale numerical features (MonthlyCharges, TotalCharges) using StandardScaler

Feature engineering: Create TenureGroups, interaction terms like MonthlyCharges * tenure

Tools:

pandas, numpy, sklearn.preprocessing

## 4. 📊 Exploratory Data Analysis (EDA)

Key Tasks:

Class imbalance: Check churn ratio

Visualize:

Churn rate by contract type, tenure, monthly charges

Histograms, boxplots, bar charts

Correlation heatmap of numeric features

Identify top potential churn drivers visually

Tools:


matplotlib, seaborn, pandas-profiling or ydata-profiling

## 5. 🧠 Model Building

Steps:

Train/Test split (70/30)

Models to try:

Logistic Regression (baseline)

Random Forest

XGBoost or LightGBM

Use GridSearchCV for hyperparameter tuning

Apply SMOTE (optional) for class imbalance

Tools:

python
Copier
Modifier
sklearn.linear_model, sklearn.ensemble, xgboost, imblearn

## 6. 📈 Model Evaluation

Metrics:

Confusion matrix

Precision, Recall, F1-Score (focus on Recall)

ROC Curve & AUC

Feature importance plot (from Random Forest/XGBoost)

Tools:

python
Copier
Modifier
sklearn.metrics, matplotlib

## 7. 🚀 Deployment Simulation (Optional Extension)

Minimal simulation:

Save model using joblib

Create a simple FastAPI or Flask app with one endpoint /predict to take customer features and return churn prediction

Tools:

joblib, FastAPI, uvicorn

## 8. 📢 Communication

Deliverables:

A final Jupyter Notebook or Python script with:

Clear section titles

Comments explaining steps

Charts with interpretations

A short summary slide or markdown:

Churn rate

Top 3 churn drivers

Confusion matrix

Model performance

🚧 BONUS (if time allows)
Build a dashboard in Streamlit or Power BI showing:

Customer churn risk

Key KPIs

Filters for customer segments

🗂️ Project Structure (Suggested)
kotlin
Copier
Modifier
project_churn_prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
├── scripts/
│   └── churn_predictor.py
├── app/
│   └── fastapi_app.py
├── requirements.txt
└── README.md