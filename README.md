

# 🚗 Multiple Linear Regression Calculator Using Car Dataset

## 📂 Directory Structure

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


## 🔧 Prerequisites
Ensure the following Python libraries are installed:
- 📦 `pandas`
- 📦 `statsmodels`

Install them via:
```bash
pip install pandas statsmodels


Install them via:
```bash
pip install pandas statsmodels
```



##📜 Scripts Overview

##📊 regression_analysis.py

**🔍 Purpose**: Perform multiple linear regression on the mtcars dataset to predict mpg (miles per gallon).

    📌 Steps: 1️⃣ Load and clean the dataset.
    2️⃣ Ensure mpg is present as the dependent variable.
    3️⃣ Fit regression models for each numeric predictor.
    4️⃣ Display results sorted by Adjusted R².

##📈 Output:

    📊 Table of predictors with:
    📉 Adjusted R²
    📌 P-value
    🧩 Coefficient
    🏎️ vif.py
    🔍 Purpose: Detect multicollinearity using Variance Inflation Factor (VIF) and evaluate predictor combinations.

##📌 Steps: 1️⃣ Load and clean the dataset.
##2️⃣ Select a primary predictor (default: wt).
##3️⃣ Calculate VIF for each predictor.
##4️⃣ Fit regression models with the primary predictor and others to identify the best combination.

##📊 Output:

    🧐 VIF Results: Table showing VIF values for predictors.
    📈 Model Evaluation: Table with Adjusted R², p-value, and coefficients for each predictor combination.
        🚀 This project helps identify the best predictors for car mileage (mpg) while addressing multicollinearity issues.



##💡 Use it to explore Multiple Linear Regression concepts and improve your statistical modeling skills!






