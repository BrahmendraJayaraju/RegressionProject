import pandas as pd

# Load the uploaded Excel file
file_path = '/Users/brahmendrajayaraju/Desktop/Book1.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
# print(data.head())

# Calculate the CacheCycleRatio as cachemem / McycTime
data['CacheCycleRatio'] = data['cachemem'] / data['McycTime']

# Display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set the display width to fit data

# Move the 'perfo' column to the end of the dataset
columns = [col for col in data.columns if col != 'perfo'] + ['perfo']
data = data[columns]

# Display the updated dataset with 'perfo' moved to the end
print(data)

from scipy.stats import boxcox, zscore
import statsmodels.api as sm

# Fit a linear model
X = data[['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']]
y = data['perfo']
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()

# Collect Box-Cox lambdas for all variables
lambda_values = {}
shifts = {}
transformed_X = X.copy()

for column in X.columns[1:]:  # Skip the intercept
    if (X[column] <= 0).any():
        # Shift data if necessary
        shift_value = abs(X[column].min()) + 1
        shifts[column] = shift_value
        shifted_data = X[column] + shift_value
    else:
        shifts[column] = 0
        shifted_data = X[column]

    # Apply Box-Cox transformation
    transformed_X[column], lambda_values[column] = boxcox(shifted_data)

# Apply Box-Cox to the response variable
if (y <= 0).any():
    response_shift = abs(y.min()) + 1
    y_shifted = y + response_shift
else:
    response_shift = 0
    y_shifted = y

y_transformed, response_lambda = boxcox(y_shifted)

# Print results
print("Lambda values for predictors:")
for k, v in lambda_values.items():
    print(f"{k}: {v}")

print(f"\nResponse variable lambda: {response_lambda}")

import numpy as np

# Apply the transformations as specified
transformed_data = data.copy()

# Applying specified transformations
transformed_data['McycTime'] = np.log(transformed_data['McycTime'])
transformed_data['minMaiMem'] = np.log(transformed_data['minMaiMem'])
transformed_data['maxMaiMem'] = np.log(transformed_data['maxMaiMem'])
transformed_data['cachemem'] = np.log(transformed_data['cachemem'])
transformed_data['minchan'] = 1 / np.sqrt(transformed_data['minchan'])
transformed_data['maxchan'] = np.log(transformed_data['maxchan'])
transformed_data['CacheCycleRatio'] = 1 / (transformed_data['CacheCycleRatio'] ** 2)
transformed_data['perfo'] = 1 / np.sqrt(transformed_data['perfo'])




transformed_data.rename(columns={
    'McycTime': 'Log(McycTime)',
    'minMaiMem': 'Log(minMaiMem)',
    'maxMaiMem': 'Log(maxMaiMem)',
    'cachemem': 'Log(cachemem)',
    'minchan': 'Inv(Sqrt(minchan))',
    'maxchan': 'Log(maxchan)',
    'CacheCycleRatio': 'Inv(Sq(CacheCycleRatio))',
    'perfo': 'Inv(Sqrt(perfo))'
}, inplace=True)

print(transformed_data.columns)


import matplotlib.pyplot as plt

# assumption 1a residual vs fitted after
# Step 2: Clean data (remove NaN or infinite values caused by transformations)
transformed_data_clean = transformed_data.replace([np.inf, -np.inf], np.nan).dropna()
predictors = ['Log(McycTime)', 'Log(minMaiMem)', 'Log(maxMaiMem)', 'Log(cachemem)', 'Inv(Sqrt(minchan))', 'Log(maxchan)', 'Inv(Sq(CacheCycleRatio))']
X = transformed_data_clean[predictors]
X_with_constant = sm.add_constant(X)

# Fit the model using the already transformed and cleaned data
y_transformed = transformed_data_clean['Inv(Sqrt(perfo))']
X_transformed = transformed_data_clean[predictors]
X_transformed_with_const = sm.add_constant(X_transformed)

model = sm.OLS(y_transformed, X_transformed_with_const).fit()

# Get fitted values and residuals
fitted_values = model.fittedvalues
residuals = model.resid

# Plot Residuals vs Fitted values
plt.figure(figsize=(18, 10))
plt.scatter(fitted_values, residuals, alpha=0.7, marker='x',color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals vs Fitted Values (Transformed Data)",fontsize=16, fontweight='bold')
plt.xlabel("Fitted Values",fontsize=16, fontweight='bold')
plt.ylabel("Residuals",fontsize=16, fontweight='bold')
plt.grid(True)
plt.show()

model_transformed = sm.OLS(y_transformed, X_transformed_with_const).fit()
# Get fitted values and residuals
fitted_values_transformed = model_transformed.fittedvalues
residuals_transformed = model_transformed.resid

# single last
# Fit a quadratic polynomial (U-shaped curve) to the residuals
coeffs = np.polyfit(fitted_values_transformed, residuals_transformed, deg=2)
poly_curve = np.poly1d(coeffs)

# Generate data points for the fitted curve
x_curve = np.linspace(min(fitted_values_transformed), max(fitted_values_transformed), 200)
y_curve = poly_curve(x_curve)

# Plot Residuals vs. Fitted Values with U-shaped curve (quadratic)
plt.figure(figsize=(18, 10))
plt.scatter(fitted_values_transformed, residuals_transformed, alpha=0.7, label="Residuals",marker='x',color='purple')
plt.plot(x_curve, y_curve, color='black', label="Quadratic Curve (U-shaped)", linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")
plt.title("Residuals vs Fitted Values with U-Shaped Quadratic Curve",fontsize=16, fontweight='bold')
plt.xlabel("Fitted Values",fontsize=16, fontweight='bold')
plt.ylabel("Residuals",fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

# individual

for predictor in predictors:
    # Extract the predictor's values and residuals
    predictor_values = transformed_data_clean[predictor]
    residuals_transformed = model_transformed.resid

    # Fit a quadratic polynomial (U-shaped curve) to the residuals
    coeffs = np.polyfit(predictor_values, residuals_transformed, deg=2)
    poly_curve = np.poly1d(coeffs)

    # Generate data points for the fitted curve
    x_curve = np.linspace(min(predictor_values), max(predictor_values), 200)
    y_curve = poly_curve(x_curve)

    # Plot Residuals vs Predictor with U-shaped curve (quadratic)
    plt.figure(figsize=(18, 10))
    plt.scatter(predictor_values, residuals_transformed, alpha=0.7, label="Residuals",marker='x',color='purple')
    plt.plot(x_curve, y_curve, color='black', label="Quadratic Curve (U-shaped)", linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")
    plt.title(f"Residuals vs {predictor} with U-Shaped Quadratic Curve",fontsize=16, fontweight='bold')
    plt.xlabel(predictor,fontsize=16, fontweight='bold')
    plt.ylabel("Residuals",fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

# Residuals vs Order plot
plt.figure(figsize=(18, 10))

# Order of residuals based on their index
residual_order = range(len(residuals_transformed))

plt.plot(residual_order, residuals_transformed,  linestyle='', alpha=0.7, label="Residuals",marker='x',color='purple')
plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")

plt.title("Residuals vs Order",fontsize=16, fontweight='bold')
plt.xlabel("Order",fontsize=16, fontweight='bold')
plt.ylabel("Residuals",fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend()
plt.show()

# influential points
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Recompute the influence measures and Bonferroni p-values
influence = model.get_influence()

# Extract Cook's Distance
cooks_d, p_values = influence.cooks_distance

# Thresholds for diagnostics
cooks_d_threshold = 4 / len(y_transformed)  # Typical Cook's Distance threshold
std_residuals_threshold = 2  # Common threshold for standardized residuals
leverage_threshold = 2 * (X_transformed_with_const.shape[1] / len(y_transformed))  # Leverage threshold

# Re-plot diagnostic plots with corrected Bonferroni p-values
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cook's Distance
sns.scatterplot(x=np.arange(len(cooks_d)), y=cooks_d, ax=axes[0, 0], color='blue')
axes[0, 0].axhline(y=cooks_d_threshold, color='red', linestyle='--', label=f"Threshold ({cooks_d_threshold:.3f})")
axes[0, 0].set_title("Cook's Distance",fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Index',fontsize=16)
axes[0, 0].set_ylabel("Cook's Distance",fontsize=16)
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot 2: Standardized Residuals
standardized_residuals = influence.resid_studentized_internal
sns.scatterplot(x=np.arange(len(standardized_residuals)), y=standardized_residuals, ax=axes[0, 1], color='green')
axes[0, 1].axhline(y=std_residuals_threshold, color='red', linestyle='--',
                   label=f"Upper Threshold ({std_residuals_threshold})")
axes[0, 1].axhline(y=-std_residuals_threshold, color='red', linestyle='--',
                   label=f"Lower Threshold (-{std_residuals_threshold})")
axes[0, 1].set_title('Standardized Residuals',fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Index',fontsize=16)
axes[0, 1].set_ylabel('Standardized Residuals',fontsize=16)
axes[0, 1].grid(True)
axes[0, 1].legend(loc='lower right')

# Plot 3: Hat Values (Leverage)
hat_values = influence.hat_matrix_diag
sns.scatterplot(x=np.arange(len(hat_values)), y=hat_values, ax=axes[1, 0],color='orange')
axes[1, 0].axhline(y=leverage_threshold, color='red', linestyle='--', label=f"Threshold ({leverage_threshold:.3f})")
axes[1, 0].set_title('Hat Values (Leverage)',fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Index',fontsize=16)
axes[1, 0].set_ylabel('Hat Values',fontsize=16)
axes[1, 0].grid(True)
axes[1, 0].legend()

# Correct Bonferroni p-values (for multiple comparisons)
bonferroni_p_values = p_values

# Plot 4: Bonferroni p-values
sns.scatterplot(x=np.arange(len(bonferroni_p_values)), y=bonferroni_p_values, ax=axes[1, 1], color='purple')

axes[1, 1].set_title('Bonferroni p-values',fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Index',fontsize=16)
axes[1, 1].set_ylabel('Bonferroni p-value',fontsize=16)
axes[1, 1].grid(True)
plt.tight_layout()
plt.show()

# this is after removing outliners
outliers = np.where(cooks_d > cooks_d_threshold)[0]  # Indices of outliers

# Correctly remove outliers by their index values
outliers_indices = transformed_data_clean.index[outliers]  # Get the actual indices of the outliers
outliers_data = transformed_data_clean.loc[outliers_indices]
remaining_data = transformed_data_clean.drop(index=outliers_indices)

# Display outliers and remaining data
print("\nremoved outliers\n", outliers_data)
# print(remaining_data)


#  after removal of outliners again apply all 4 4
y_transformed_out = remaining_data['Inv(Sqrt(perfo))']
X_transformed_out = remaining_data[predictors]
X_transformed_with_const_out = sm.add_constant(X_transformed_out)

model_out = sm.OLS(y_transformed_out, X_transformed_with_const_out).fit()
# Recompute the influence measures and Bonferroni p-values
influence_out = model_out.get_influence()

# Extract Cook's Distance
cooks_d_out, p_values_out = influence_out.cooks_distance

cooks_d_threshold_out = 4 / len(y_transformed_out)  # Typical Cook's Distance threshold
std_residuals_threshold_out = 2  # Common threshold for standardized residuals
leverage_threshold_out = 2 * (X_transformed_with_const_out.shape[1] / len(y_transformed_out))  # Leverage threshold

# Re-plot diagnostic plots with corrected Bonferroni p-values
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cook's Distance
sns.scatterplot(x=np.arange(len(cooks_d_out)), y=cooks_d_out, ax=axes[0, 0], color='blue')
axes[0, 0].axhline(y=cooks_d_threshold_out, color='red', linestyle='--',
                   label=f"Threshold ({cooks_d_threshold_out:.3f})")
axes[0, 0].set_title("Cook's Distance",fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Index',fontsize=16)
axes[0, 0].set_ylabel("Cook's Distance",fontsize=16)
axes[0, 0].grid(True)
axes[0, 0].legend()

# Plot 2: Standardized Residuals
standardized_residuals_out = influence_out.resid_studentized_internal
sns.scatterplot(x=np.arange(len(standardized_residuals_out)), y=standardized_residuals_out, ax=axes[0, 1],
                color='green')
axes[0, 1].axhline(y=std_residuals_threshold_out, color='red', linestyle='--',
                   label=f"Upper Threshold ({std_residuals_threshold_out})")
axes[0, 1].axhline(y=-std_residuals_threshold_out, color='red', linestyle='--',
                   label=f"Lower Threshold (-{std_residuals_threshold_out})")
axes[0, 1].set_title('Standardized Residuals',fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Index',fontsize=16)
axes[0, 1].set_ylabel('Standardized Residuals',fontsize=16)
axes[0, 1].grid(True)
axes[0, 1].legend(loc='lower right')

# Plot 3: Hat Values (Leverage)
hat_values_out = influence_out.hat_matrix_diag
sns.scatterplot(x=np.arange(len(hat_values_out)), y=hat_values_out, ax=axes[1, 0], color='orange')
axes[1, 0].axhline(y=leverage_threshold_out, color='red', linestyle='--',
                   label=f"Threshold ({leverage_threshold_out:.3f})")
axes[1, 0].set_title('Hat Values (Leverage)',fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Index',fontsize=16)
axes[1, 0].set_ylabel('Hat Values',fontsize=16)
axes[1, 0].grid(True)
axes[1, 0].legend()

# Correct Bonferroni p-values (for multiple comparisons)
bonferroni_p_values_out = p_values_out

# Plot 4: Bonferroni p-values
sns.scatterplot(x=np.arange(len(bonferroni_p_values_out)), y=bonferroni_p_values_out, ax=axes[1, 1], color='purple')

axes[1, 1].set_title('Bonferroni p-values',fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Index',fontsize=16)
axes[1, 1].set_ylabel('Bonferroni p-value',fontsize=16)
axes[1, 1].grid(True)
plt.tight_layout()
plt.show()

# QQ plot # Create a final QQ plot for the transformed response variable 'perfo'

import scipy.stats as stats
import matplotlib.pyplot as plt

# Create QQ plot
plt.figure(figsize=(18, 10))
res = stats.probplot(remaining_data['Inv(Sqrt(perfo))'], dist="norm")
# Plot the scatter points with 'x' markers
plt.scatter(res[0][0], res[0][1], marker='x', color='purple', label='Data Points')
# Plot the regression line in black
plt.plot(res[0][0], res[1][0] * res[0][0] + res[1][1], color='black', label='Regression Line')
# Customize titles and labels
plt.title("QQ Plot for Transformed Response Variable ('Inv(Sqrt(perfo))')", fontsize=16, fontweight='bold')
plt.xlabel("Theoretical Quantiles", fontsize=16, fontweight='bold')
plt.ylabel("Sample Quantiles", fontsize=16, fontweight='bold')
# Add grid and legend
plt.grid(True)
plt.legend(fontsize=12)
plt.show()

# Plot a histogram for the transformed response variable 'perfo'
plt.figure(figsize=(18, 10))
plt.hist(remaining_data['Inv(Sqrt(perfo))'], bins=15,color='purple', edgecolor='black', alpha=0.7)
plt.title("Histogram of Transformed Response Variable ('Inv(Sqrt(perfo))')",fontsize=16, fontweight='bold')
plt.xlabel("Transformed perfo",fontsize=16, fontweight='bold')
plt.ylabel("Frequency",fontsize=16, fontweight='bold')
plt.grid(axis='y', alpha=0.75)
plt.show()
