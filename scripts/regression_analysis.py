import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
data = pd.read_csv('../data/mtcars.csv')

# Define the independent variables (predictors)
X = data[['wt', 'qsec', 'cyl', 'hp']]
X = sm.add_constant(X)  # Add a constant term for the intercept

# Define the dependent variable (target)
y = data['mpg']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Calculate Variance Inflation Factor (VIF) for each predictor
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# Save VIF to a CSV
vif_data.to_csv('../results/vif.csv', index=False)

# Save regression summary to a text file
with open('../results/regression_summary.txt', 'w') as f:
    f.write(model.summary().as_text())

# Save residuals plot (optional)
import matplotlib.pyplot as plt

plt.scatter(model.fittedvalues, model.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.savefig('../results/residual_plot.png')
plt.close()

