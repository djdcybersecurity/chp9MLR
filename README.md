

# Multiple Linear Regression Calculator Using Car Dataset

## Directory Structure
```
chp9MLR/
├── results/
├── scripts/
│   ├── regression_analysis.py
│   ├── vif.py
│   ├── mtcars.csv
├── .gitignore
├── README.md
```

## Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `statsmodels`

Install them via:
```bash
pip install pandas statsmodels
```

## Scripts Overview

### `regression_analysis.py`
**Purpose**: Perform multiple linear regression on the `mtcars` dataset to predict `mpg` (miles per gallon).

**Steps**:
1. Load and clean the dataset.
2. Ensure `mpg` is present as the dependent variable.
3. Fit regression models for each numeric predictor.
4. Display results sorted by adjusted R-squared.

**Output**:
- Table of predictors with:
  - **Adjusted R²**
  - **P-value**
  - **Coefficient**

---

### `vif.py`
**Purpose**: Detect multicollinearity using Variance Inflation Factor (VIF) and evaluate predictor combinations.

**Steps**:
1. Load and clean the dataset.
2. Select a primary predictor (`wt` by default).
3. Calculate VIF for each predictor.
4. Fit regression models with the primary predictor and others to identify the best combination.

**Output**:
- **VIF Results**: Table showing VIF values for predictors.
- **Model Evaluation**: Table with adjusted R², p-value, and coefficients for each predictor combination.

---

This version retains all necessary information while removing redundancy and simplifying the language.



