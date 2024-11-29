import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from partD import data
import matplotlib.pyplot as plt

# Define the response variable and predictors
remaining_data_Model = data.rename(columns={
    "Log(McycTime)": "Log_McycTime",
    "Log(minMaiMem)": "Log_minMaiMem",
    "Log(maxMaiMem)": "Log_maxMaiMem",
    "Log(cachemem)": "Log_cachemem",
    "Log(maxchan)": "Log_maxchan",
    "Inv(Sqrt(perfo))": "Inv_Sqrt_perfo"
})
response = 'Inv_Sqrt_perfo'
predictors = ['Log_McycTime', 'Log_minMaiMem', 'Log_maxMaiMem', 'Log_cachemem', 'Log_maxchan']
data = remaining_data_Model

print(data)

# Generate interaction terms and add to the dataset
interaction_terms = []
for combo in combinations(predictors, 2):
    interaction_col = f"{combo[0]}_x_{combo[1]}"
    data[interaction_col] = data[combo[0]] * data[combo[1]]
    interaction_terms.append(interaction_col)

# Combine original predictors and interaction terms
all_predictors = predictors + interaction_terms

print(all_predictors)

# Fit the full model with interactions
X_full = sm.add_constant(data[all_predictors])
y = data[response]
remaining_predictors = list(X_full.columns)

# Initialize variables for tracking backward elimination
steps = []
bic_values = []

while len(remaining_predictors) > 1:
    # Fit the model with the current predictors
    X_temp = X_full[remaining_predictors]
    model_temp = sm.OLS(y, X_temp).fit()

    # Record current step information
    bic_values.append(model_temp.bic)
    steps.append({
        "Remaining Predictors": remaining_predictors.copy(),
        "BIC": model_temp.bic,
    })

    # Identify the predictor with the highest p-value
    p_values = model_temp.pvalues
    worst_predictor = p_values.idxmax()

    # Break the loop if all predictors are significant
    if p_values[worst_predictor] < 0.05:
        break

    # Remove the worst predictor and record its removal
    remaining_predictors.remove(worst_predictor)
    steps[-1]["Removed Predictor"] = worst_predictor

# Convert steps into a DataFrame for display
step_df = pd.DataFrame(steps)
step_df["Step"] = range(1, len(step_df) + 1)

# Fit the final model
final_model = sm.OLS(y, X_full[remaining_predictors]).fit()

# Display intermediate steps and final model predictors
print("\nBackward Elimination Steps:")
print(step_df)
print("\nFinal Model Predictors:", remaining_predictors)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for the final model predictors
X_final = X_full[remaining_predictors]  # Subset the design matrix for final predictors
vif_data = pd.DataFrame({
    "Predictor": X_final.columns,
    "VIF": [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
})

print("\nVariance Inflation Factor (VIF) for Final Model Predictors:")
print(vif_data)

# Calculate VIF for the final model predictors


X_final = X_full[remaining_predictors]  # Subset the design matrix for final predictors
columns_to_remove = ['Log_McycTime_x_Log_minMaiMem', 'Log_minMaiMem_x_Log_maxMaiMem', 'Log_minMaiMem_x_Log_maxchan', 'Log_McycTime_x_Log_maxchan',
                     'Log_cachemem_x_Log_maxchan']
X_final = X_final.drop(columns=columns_to_remove)
print(X_final)

vif_data = pd.DataFrame({
    "Predictor": X_final.columns,
    "VIF": [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
})

print("\nVariance Inflation Factor (VIF) for Final Model Predictors:")
print(vif_data)

# start assumptions
X_final = data.copy()
X_final['response'] = data[response]

predictors = ['Log_minMaiMem', 'Log_maxMaiMem', 'Log_maxchan', 'Log_McycTime_x_Log_maxMaiMem', 'Log_McycTime_x_Log_cachemem']
X = X_final[predictors]

transformed_data_clean_2model = X_final.replace([np.inf, -np.inf], np.nan).dropna()
X_with_constant = sm.add_constant(X)
# Fit the model using the already transformed and cleaned data
y_transformed = transformed_data_clean_2model['Inv_Sqrt_perfo']
X_transformed = transformed_data_clean_2model[predictors]
X_transformed_with_const = sm.add_constant(X_transformed)

model = sm.OLS(y_transformed, X_transformed_with_const).fit()

# Get fitted values and residuals
fitted_values = model.fittedvalues
residuals = model.resid

# Plot Residuals vs Fitted values
plt.figure(figsize=(18, 10))
plt.scatter(fitted_values, residuals, alpha=0.7,marker='x',color='purple')
plt.axhline(y=0, color='red', linestyle='--')

plt.title("Residuals vs Fitted Values  -Model 2 ",fontsize=16, fontweight='bold')
plt.xlabel("Fitted Values",fontsize=16, fontweight='bold')
plt.ylabel("Residuals",fontsize=16, fontweight='bold')
plt.grid(True)
plt.show()

model_transformed = sm.OLS(y_transformed, X_transformed_with_const).fit()
fitted_values_transformed = model_transformed.fittedvalues
residuals_transformed = model_transformed.resid

# with U shaped curve
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
plt.title("Residuals vs Fitted Values with U-Shaped Quadratic Curve -Model 2",fontsize=16, fontweight='bold')
plt.xlabel("Fitted Values",fontsize=16, fontweight='bold')
plt.ylabel("Residuals",fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()

# individual

for predictor in predictors:
    # Extract the predictor's values and residuals
    predictor_values = transformed_data_clean_2model[predictor]
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
    plt.title(f"Residuals vs {predictor} with U-Shaped Quadratic Curve -Model 2",fontsize=16, fontweight='bold')
    plt.xlabel(predictor,fontsize=16, fontweight='bold')
    plt.ylabel("Residuals",fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()

# Residuals vs Order plot
plt.figure(figsize=(18, 10))

# Order of residuals based on their index
residual_order = range(len(residuals_transformed))
plt.plot(residual_order, residuals_transformed, linestyle='', alpha=0.7, label="Residuals",marker='x',color='purple')
plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")
plt.title("Residuals vs Order -Model 2 ",fontsize=16, fontweight='bold')
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
axes[0, 0].set_title("Cook's Distance -Model 2 ",fontsize=16, fontweight='bold')
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
axes[0, 1].set_title('Standardized Residuals -Model 2',fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Index',fontsize=16)
axes[0, 1].set_ylabel('Standardized Residuals',fontsize=16)
axes[0, 1].grid(True)
axes[0, 1].legend(loc='lower right')

# Plot 3: Hat Values (Leverage)
hat_values = influence.hat_matrix_diag
sns.scatterplot(x=np.arange(len(hat_values)), y=hat_values, ax=axes[1, 0], color='orange')
axes[1, 0].axhline(y=leverage_threshold, color='red', linestyle='--', label=f"Threshold ({leverage_threshold:.3f})")
axes[1, 0].set_title('Hat Values (Leverage) -Model 2',fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Index',fontsize=16)
axes[1, 0].set_ylabel('Hat Values',fontsize=16)
axes[1, 0].grid(True)
axes[1, 0].legend()

# Correct Bonferroni p-values (for multiple comparisons)
bonferroni_p_values = p_values

# Plot 4: Bonferroni p-values
sns.scatterplot(x=np.arange(len(bonferroni_p_values)), y=bonferroni_p_values, ax=axes[1, 1], color='purple')

axes[1, 1].set_title('Bonferroni p-values -Model 2',fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Index',fontsize=16)
axes[1, 1].set_ylabel('Bonferroni p-value',fontsize=16)
axes[1, 1].grid(True)
plt.tight_layout()
plt.show()



# QQ plot # Create a final QQ plot for the transformed response variable 'perfo'

import scipy.stats as stats
import matplotlib.pyplot as plt



plt.figure(figsize=(18, 10))
res = stats.probplot(transformed_data_clean_2model['Inv_Sqrt_perfo'], dist="norm")
# Plot the scatter points with 'x' markers
plt.scatter(res[0][0], res[0][1], marker='x', color='purple', label='Data Points')
# Plot the regression line in black
plt.plot(res[0][0], res[1][0] * res[0][0] + res[1][1], color='black', label='Regression Line')
# Customize titles and labels
plt.title("QQ Plot for Transformed Response Variable (Inv_Sqrt_perfo) -Model 2 ",fontsize=16, fontweight='bold')
plt.xlabel("Theoretical Quantiles", fontsize=16, fontweight='bold')
plt.ylabel("Sample Quantiles", fontsize=16, fontweight='bold')
# Add grid and legend
plt.grid(True)
plt.legend(fontsize=12)
plt.show()









# Plot a histogram for the transformed response variable 'perfo'
plt.figure(figsize=(18, 10))
plt.hist(transformed_data_clean_2model['Inv_Sqrt_perfo'], bins=15, color='purple',edgecolor='black', alpha=0.7)
plt.title("Histogram of Transformed Response Variable (Inv_Sqrt_perfo) -Model 2 ",fontsize=16, fontweight='bold')
plt.xlabel("Transformed perfo",fontsize=16, fontweight='bold')
plt.ylabel("Frequency",fontsize=16, fontweight='bold')
plt.grid(axis='y', alpha=0.75)
plt.show()






#Anova
data =transformed_data_clean_2model
response = 'Inv_Sqrt_perfo'
predictors = ['Log_minMaiMem', 'Log_maxMaiMem', 'Log_maxchan', 'Log_McycTime_x_Log_maxMaiMem', 'Log_McycTime_x_Log_cachemem']
# Construct the formula for regression
formula = f"{response} ~ {' + '.join(predictors)}"
import statsmodels.formula.api as smf
# Fit the model
model = smf.ols(formula=formula, data=transformed_data_clean_2model ).fit()

# Obtain parameter estimates
params = model.params
# Perform type III ANOVA
anova_results = sm.stats.anova_lm(model, typ=3)
print("\nsummary of model -Model 2\n", model.summary())
# Extract standard errors and p-values from the model
standard_errors = model.bse  # Standard errors
p_values = model.pvalues  # p-values
print("\nP value seperately -Model 2\n", p_values)
print("\nSE seperately -Model 2\n", standard_errors)

# Calculate R-squared
r_squared = model.rsquared

# Determine the number of rows and columns for the grid
rows, cols = 3, 2  # 3 rows and 2 columns

# Create subplots for the effect plots
fig, axes = plt.subplots(rows, cols, figsize=(18, 18), constrained_layout=True)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot each predictor
for i, predictor in enumerate(predictors):
    sm.graphics.plot_ccpr(model, predictor, ax=axes[i])
    axes[i].set_title(f"Effect Plot for {predictor} -Model 2",fontsize=16, fontweight='bold')
    axes[i].set_xlabel(predictor,fontsize=16)
    axes[i].set_ylabel(response,fontsize=16)

# Turn off unused subplots
for j in range(len(predictors), len(axes)):
    axes[j].axis('off')

plt.suptitle("\nEffect Plots for Predictors -Model 2\n", fontsize=16,fontweight='bold')
plt.show()

# Display ANOVA table and parameter estimates
print("\nAnova Table -Model 2\n",anova_results)
print("\nParameter Estimates -Model 2 :\n")
print(params)

# Summary for R-squared and interpretation
interpretation = {
    "R_squared -Model 2": r_squared,
    "Interpretation": "The R-squared value represents the proportion of variance in the response variable 'perfo' "
                      "that is explained by the predictors in the model."
}
print("\nR^2 -Model 2 \n",interpretation)
