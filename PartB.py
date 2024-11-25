import pandas as pd

# Load the uploaded Excel file
file_path = '/Users/brahmendrajayaraju/Desktop/Book1.xlsx'
data = pd.ExcelFile(file_path)
# Load the data from the first sheet
df = data.parse('Sheet1')
# Add a new column 'CacheCycleRatio' as cachemem / McycTime
df['CacheCycleRatio'] = df['cachemem'] / df['McycTime']

# Reorder the columns to place 'CacheCycleRatio' before 'perfo'
cols = df.columns.tolist()
cols.remove('CacheCycleRatio')
cols.insert(cols.index('perfo'), 'CacheCycleRatio')

# Reorder and display the updated DataFrame
df = df[cols]

print(df)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Select predictors and response variable
predictors = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']
response = 'perfo'

# Plot histograms for predictors and response variable
fig, axes = plt.subplots(len(predictors) + 1, 1, figsize=(8, 20), tight_layout=True)
for i, predictor in enumerate(predictors):
    sns.histplot(df[predictor], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {predictor}')
axes[-1].set_title(f'Distribution of {response}')
sns.histplot(df[response], kde=True, ax=axes[-1])

plt.show()

# Check skewness of predictors and response variable
skewness = df[predictors + [response]].skew()
print(skewness)

# Apply logarithmic transformation to predictors and response variable
log_transformed_data = df.copy()

# Apply log transformation to predictors and response (adding 1 to avoid log(0))
for column in predictors + [response]:
    log_transformed_data[column] = np.log1p(log_transformed_data[column])

# Check the transformed distributions
fig, axes = plt.subplots(len(predictors) + 1, 1, figsize=(8, 20), tight_layout=True)
for i, predictor in enumerate(predictors):
    sns.histplot(log_transformed_data[predictor], kde=True, ax=axes[i])
    axes[i].set_title(f'Log-Transformed Distribution of {predictor}')
axes[-1].set_title(f'Log-Transformed Distribution of {response}')
sns.histplot(log_transformed_data[response], kde=True, ax=axes[-1])

plt.show()

# Check skewness of transformed data
log_transformed_skewness = log_transformed_data[predictors + [response]].skew()
print(log_transformed_skewness)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for the predictors in the log-transformed data
X = log_transformed_data[predictors]
X['Intercept'] = 1  # Add an intercept for the VIF calculation

# Compute VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Variable'] = predictors
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(predictors))]

# Drop the intercept column after calculation
X.drop(columns=['Intercept'], inplace=True)

print(vif_data)

# part 1 Linearity: check
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import Polynomial
# Define predictors and response
predictors = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']
response = 'perfo'

# Prepare the design matrix (with an intercept)
X = sm.add_constant(df[predictors])
y = df[response]

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Calculate residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Create residual plot
plt.figure(figsize=(8, 6))
p_original = Polynomial.fit(fitted_values, residuals, deg=3)  # Fit a cubic curve
x_curve = np.linspace(min(fitted_values), max(fitted_values), 100)
y_curve = p_original(x_curve)
plt.plot(x_curve, y_curve, color="blue", label="Fitted Curve", linewidth=2)
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1.2)
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Display model summary to understand fit
model_summary = model.summary()
print(model_summary)

#remove using log transform
# Attempt to improve the model by applying log transformations to predictors and response
log_transformed_data = df.copy()

# Log-transform predictors and response to address non-linearity and stabilize variance (adding 1 to avoid log(0))
predictors_with_log = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']
response = 'perfo'

for column in predictors_with_log + [response]:
    log_transformed_data[column] = np.log1p(log_transformed_data[column])

# Fit the linear regression model with log-transformed data
X_log = sm.add_constant(log_transformed_data[predictors_with_log])
y_log = log_transformed_data[response]
log_model = sm.OLS(y_log, X_log).fit()

# Calculate residuals and fitted values for the transformed model
residuals_log = log_model.resid
fitted_values_log = log_model.fittedvalues

# Plot residuals vs. fitted values for the log-transformed model
plt.figure(figsize=(8, 6))
p_log = Polynomial.fit(fitted_values_log, residuals_log, deg=3)
x_curve_log = np.linspace(min(fitted_values_log), max(fitted_values_log), 100)
y_curve_log = p_log(x_curve_log)
plt.plot(x_curve_log, y_curve_log, color="orange", label="Fitted Curve", linewidth=2)
plt.scatter(fitted_values_log, residuals_log, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1.2)
plt.title('Residuals vs. Fitted Values (Log-Transformed Data)')
plt.xlabel('Fitted Values (Log-Transformed)')
plt.ylabel('Residuals')
plt.show()

# Display summary of the log-transformed model
log_model_summary = log_model.summary()
print(log_model_summary)




#PART 2 RESIDUAL VS ORDER
import numpy as np
df['CacheCycleRatio'] = df['cachemem'] / df['McycTime']

# Recalculate residuals for the original model
X_original = sm.add_constant(df[predictors])
y_original = df[response]
model_original = sm.OLS(y_original, X_original).fit()
residuals_original = model_original.resid
# Log-transform predictors and response
log_transformed_data = df.copy()
for column in predictors + [response]:
    log_transformed_data[column] = np.log1p(log_transformed_data[column])



# Recalculate residuals for the log-transformed model
X_log_transformed = sm.add_constant(log_transformed_data[predictors])
y_log_transformed = log_transformed_data[response]
model_log_transformed = sm.OLS(y_log_transformed, X_log_transformed).fit()
residuals_log_transformed = model_log_transformed.resid



# Residuals vs. Order Plot for the Original Model
plt.figure(figsize=(8, 6))
plt.plot(range(len(residuals_original)), residuals_original, marker='o', linestyle='', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2)
plt.title('Residuals vs. Order (Original Model)')
plt.xlabel('Order of Observations')
plt.ylabel('Residuals')
plt.show()

# Residuals vs. Order Plot for the Log-Transformed Model
plt.figure(figsize=(8, 6))
plt.plot(range(len(residuals_log_transformed)), residuals_log_transformed, marker='o', linestyle='', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2)
plt.title('Residuals vs. Order (Log-Transformed Model)')
plt.xlabel('Order of Observations')
plt.ylabel('Residuals')
plt.show()
















#part 4 COOKS DISTANCE PENDING








#part 5 normality of resideuals



# Q-Q plot for the residuals of the original model
plt.figure(figsize=(8, 6))
sm.qqplot(residuals_original, line='45', fit=True)
plt.title('Q-Q Plot of Residuals (Original Model)')
plt.show()

# Q-Q plot for the residuals of the log-transformed model
plt.figure(figsize=(8, 6))
sm.qqplot(residuals_log_transformed, line='45', fit=True)
plt.title('Q-Q Plot of Residuals (Log-Transformed Model)')
plt.show()
