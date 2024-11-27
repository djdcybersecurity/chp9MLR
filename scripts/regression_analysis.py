import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# Load dataset
mtcars = pd.read_csv("path_to_mtcars.csv")

# Select variables
mtcars = mtcars[['mpg', 'wt', 'qsec', 'cyl', 'hp']]

# Prepare data for regression
X = mtcars[['wt', 'qsec', 'cyl', 'hp']]
y = mtcars['mpg']
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# R² and Adjusted R²
print("R²:", model.rsquared)
print("Adjusted R²:", model.rsquared_adj)

# Confidence intervals
print("Confidence Intervals:")
print(model.conf_int())

# Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Residual diagnostics
residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.subplot(1, 2, 2)
plt.hist(residuals, bins=15, edgecolor='k', alpha=0.7)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
