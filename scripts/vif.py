import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
try:
    mtcars = pd.read_csv('mtcars.csv')  # Ensure file exists in the current working directory
except FileNotFoundError:
    raise FileNotFoundError("The 'mtcars.csv' file was not found. Ensure it is in the current directory.")

# Drop car names (if present), ensure numeric data, and drop rows with missing values
if 'Unnamed: 0' in mtcars.columns:
    mtcars = mtcars.drop(columns=['Unnamed: 0'])
mtcars = mtcars.apply(pd.to_numeric, errors='coerce').dropna()

# Ensure 'mpg' exists in the dataset
if 'mpg' not in mtcars.columns:
    raise ValueError("Dataset must include 'mpg' as the dependent variable.")

# Define the dependent variable
y = mtcars['mpg']

# Step 1: Select the best predictor based on prior analysis
# Initial best predictor selection (hardcoded or update based on your analysis)
best_predictor = 'wt'

# Step 2: Identify remaining predictors excluding 'mpg' and the best predictor
remaining_predictors = [
    col for col in mtcars.columns if col != 'mpg' and col != best_predictor and pd.api.types.is_numeric_dtype(mtcars[col])
]

# Check if there are any predictors left
if not remaining_predictors:
    raise ValueError("No additional predictors available after excluding 'mpg' and the best predictor.")

# Calculate VIF to check multicollinearity among remaining predictors
X_vif = sm.add_constant(mtcars[remaining_predictors])  # Add constant for VIF calculation
vif_data = pd.DataFrame()
vif_data["Predictor"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\nVIF Results (for multicollinearity):")
print(vif_data)

# Step 3: Fit models with each remaining predictor and evaluate adjusted R-squared
results = []
for predictor in remaining_predictors:
    X = mtcars[[best_predictor, predictor]]  # Model with the best predictor and one additional predictor
    X = sm.add_constant(X)  # Add constant for intercept
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Append model evaluation metrics
    results.append({
        'Predictor': predictor,
        'Adj_R2': model.rsquared_adj,
        'P_Value': model.pvalues.get(predictor, None),  # Use .get() to avoid index errors
        'Coefficient': model.params.get(predictor, None),  # Use .get() to avoid index errors
    })

# Convert results to a DataFrame and display sorted by Adjusted R^2
results_df = pd.DataFrame(results).sort_values(by='Adj_R2', ascending=False)
results_df = results_df.round({'Adj_R2': 4, 'P_Value': 4, 'Coefficient': 4})  # Format values
print("\nSecond-best predictor model results:")
print(results_df)

