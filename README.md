# MLOps Introduction: Final Project

## Student Information
- Full name: Paulo Sucasaire
- Email: paulo.morales.s@uni.pe
- Grupo: 1 (2026)

---

# Project Title
## Supervised Learning Models to Predict Diabetes Risk in Peruvian Adults (ENDES 2020–2024)

---

# 1. Problem Definition

This project aims to predict whether an adult Peruvian (≥18 years old) has been diagnosed with diabetes, using demographic, socioeconomic, and lifestyle variables from the ENDES survey.

### Business / Public Health Motivation
Diabetes is a major chronic disease in Peru. Identifying individuals at high risk allows:

- Early intervention
- Preventive policies
- Resource prioritization

### ML Formulation

- **Task:** Binary Classification
- **Target:** `diabetes` (1 = diagnosed, 0 = not diagnosed)
- **Dataset:** ENDES 2020–2024
- **Observations:** 148,010 adults
- **Class imbalance:** ~3.7% positive class

---

# 2. Project Structure
uni_mds_mlops-final-project/
│
├── data/
│ ├── raw/
│ └── training/
│
├── models/
│
├── reports/
│
├── src/
│ ├── data_preparation.py
│ ├── train.py
│ └── serving.py
│
└── README.md


---

# 3. Data Preparation

The script `src/data_preparation.py`:

- Loads raw parquet data
- Filters adults (≥18 years, complete interview)
- Creates target variable (`diabetes`)
- Engineers selected features
- Outputs a clean training dataset

### Run:

```bash
python src/data_preparation.py \
  --raw-path data/raw/endes_diabetes_model.parquet \
  --out-path data/training/endes_diabetes_features.parquet

  Output:
data/training/endes_diabetes_features.parquet
Shape: (148,010, 20)


# 4. Model Training & Champion Selection

The script `src/train.py`:

- Splits data (80/20 stratified)
- Applies preprocessing:
  - Median imputation (numerical features)
  - OneHotEncoding (categorical features)
- Trains:
  - Logistic Regression
  - Random Forest
- Evaluates:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Selects the champion model based on:
  - Highest F1 score
  - Tie-breaker: ROC-AUC

## Run Training

```bash
python src/train.py \
  --data-path data/training/endes_diabetes_features.parquet \
  --out-dir models \
  --test-size 0.2 \
  --random-state 42

# 5. Results

## Champion Model: Logistic Regression

| Metric    | Logistic Regression | Random Forest |
|-----------|--------------------|---------------|
| Accuracy  | 0.770              | 0.962         |
| Precision | 0.119              | 0.438         |
| Recall    | 0.795              | 0.006         |
| F1 Score  | 0.206              | 0.012         |
| ROC-AUC   | 0.860              | 0.861         |

## Interpretation

Although Random Forest achieved very high accuracy, it failed to correctly identify positive cases (very low recall).

Logistic Regression achieved:

- High recall (0.79)
- Balanced F1 score
- Strong ROC-AUC (0.86)

Given the public health context, recall is critical for identifying individuals at risk of diabetes.

# 6. Model Serving (FastAPI)

The trained champion model is exposed through a REST API using **FastAPI**.

The API loads the serialized model (`models/champion_model.joblib`) and provides prediction endpoints.

---

## Run the API

```bash
uvicorn src.serving:app --host 0.0.0.0 --port 8000
