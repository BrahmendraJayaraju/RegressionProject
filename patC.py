import pandas as pd
from partBRefactor import remaining_data

# Load the uploaded Excel file


file_path = '/Users/brahmendrajayaraju/Desktop/Book1.xlsx'
df = pd.read_excel(file_path)

# Create a new column "CacheCycleRatio" by calculating cachemem / McycTime
df['CacheCycleRatio'] = df['cachemem'] / df['McycTime']

# Reorder columns to place "CacheCycleRatio" before "perfo"
cols = list(df.columns)
cols.remove('CacheCycleRatio')
cols.insert(cols.index('perfo'), 'CacheCycleRatio')
df = df[cols]

print("\nremaining_data after removal\n",remaining_data)
# Prepare the predictors and response variables
predictors = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']
response = 'perfo'
X = remaining_data[predictors]
y = remaining_data[response]
from statsmodels.api import OLS, add_constant


# Forward Selection with Marginality Principle Enforcement
def forward_selection_with_marginality(X, y, significance_level=0.05):
    initial_features = []
    remaining_features = list(X.columns)
    selected_features = []

    # Dictionary to track lower-order terms required for marginality enforcement
    hierarchy = {feature: [] for feature in X.columns}

    # Example: Define hierarchical relationships (customize this for specific data)
    # If you have interaction terms, define their components
    # Example: hierarchy['X1*X2'] = ['X1', 'X2']
    # For polynomial terms, define their base terms
    # Example: hierarchy['X1^2'] = ['X1']

    step_details = []

    while remaining_features:
        p_values = {}
        for feature in remaining_features:
            # Check if adding this feature would violate marginality
            required_terms = hierarchy.get(feature, [])
            if not all(term in initial_features for term in required_terms):
                continue  # Skip this feature if marginality would be violated

            # Evaluate the model with the current feature
            model = OLS(y, add_constant(X[initial_features + [feature]])).fit()
            p_values[feature] = model.pvalues[feature]

        if not p_values:
            break  # No valid features left to add

        min_p_value = min(p_values.values())
        selected_feature = min(p_values, key=p_values.get)

        if min_p_value < significance_level:
            initial_features.append(selected_feature)
            remaining_features.remove(selected_feature)
            selected_features.append(selected_feature)

            # Automatically add required lower-order terms for marginality if missing
            for required_term in hierarchy.get(selected_feature, []):
                if required_term not in initial_features:
                    initial_features.append(required_term)
                    remaining_features.remove(required_term)

            step_details.append({
                "Step": len(step_details) + 1,
                "P-Values": p_values.copy(),
                "Selected P-Value": min_p_value,
                "Selected Feature": selected_feature,
                "Remaining Predictors": list(remaining_features),
                "Selected Predictors": list(initial_features),
            })
        else:
            break

    return selected_features, step_details


# Perform forward selection enforcing marginality
selected_features_marginality, steps_marginality = forward_selection_with_marginality(X, y)

# Print the details for each step
for step in steps_marginality:
    print(f"\nStep {step['Step']}:")
    print(f"P-Values: {step['P-Values']}")
    print(f"Selected Feature: {step['Selected Feature']} with P-Value: {step['Selected P-Value']}")
    print(f"Remaining Predictors: {step['Remaining Predictors']}")
    print(f"Selected Predictors: {step['Selected Predictors']}\n")

# Print final selected features
print("\nFinal Selected Features:\n")
print("\nselected_features_marginality\n",selected_features_marginality)

print(".........................................................................................\n")
print("\nbackward elimination with ftest\n")


# Backward Elimination with Marginality Principle Enforcement
def backward_selection_with_marginality(X, y, significance_level=0.05):
    initial_features = list(X.columns)
    selected_features = list(initial_features)

    # Dictionary to track lower-order terms required for marginality enforcement
    hierarchy = {feature: [] for feature in X.columns}

    step_details = []

    while selected_features:
        p_values = {}
        model = OLS(y, add_constant(X[selected_features])).fit()

        # Evaluate p-values for all remaining features
        for feature in selected_features:
            p_values[feature] = model.pvalues.get(feature, 1)

        max_p_value = max(p_values.values())
        worst_feature = max(p_values, key=p_values.get)

        if max_p_value > significance_level:
            # Check if removing this feature would violate marginality
            for dependent_feature, dependencies in hierarchy.items():
                if worst_feature in dependencies and dependent_feature in selected_features:
                    break
            else:
                # If no violation, remove the worst feature
                selected_features.remove(worst_feature)
                step_details.append({
                    "Step": len(step_details) + 1,
                    "P-Values": p_values.copy(),
                    "Removed P-Value": max_p_value,
                    "Removed Feature": worst_feature,
                    "Remaining Predictors": list(selected_features),
                })
        else:
            break

    return selected_features, step_details


# Perform backward elimination enforcing marginality
selected_features_backward, steps_backward = backward_selection_with_marginality(X, y)

formatted_backward_output = "\n".join([
    f"Step {step['Step']}:\n"
    f"P-Values: {step['P-Values']}\n"
    f"Removed Feature: {step['Removed Feature']} with P-Value: {step['Removed P-Value']}\n"
    f"Remaining Predictors: {step['Remaining Predictors']}\n"
    for step in steps_backward
])

# Final selected features
formatted_final_output = f"Final Selected Features:\n{selected_features_backward}"

print("\nformatted_backward_output\n",formatted_backward_output)
print("\nformatted_final_output\n",formatted_final_output)

print("\n......................................................................................\n")
print("\nforward selection with AIC\n")

from statsmodels.api import OLS, add_constant


# Forward Selection based on AIC
def forward_selection_with_aic(X, y):
    initial_features = []
    remaining_features = list(X.columns)
    selected_features = []
    step_details = []

    current_aic = float('inf')  # Start with a very high AIC

    while remaining_features:
        aic_values = {}
        for feature in remaining_features:
            # Evaluate the model with the current feature added
            model = OLS(y, add_constant(X[initial_features + [feature]])).fit()
            aic_values[feature] = model.aic

        min_aic = min(aic_values.values())
        selected_feature = min(aic_values, key=aic_values.get)

        if min_aic < current_aic:
            current_aic = min_aic
            initial_features.append(selected_feature)
            remaining_features.remove(selected_feature)
            selected_features.append(selected_feature)

            step_details.append({
                "Step": len(step_details) + 1,
                "AIC Values": aic_values.copy(),
                "Selected AIC": min_aic,
                "Selected Feature": selected_feature,
                "Remaining Predictors": list(remaining_features),
                "Selected Predictors": list(initial_features),
            })
        else:
            break

    return selected_features, step_details


# Perform forward selection based on AIC
selected_features_aic, steps_aic = forward_selection_with_aic(X, y)

# Print the details for each step
for step in steps_aic:
    print(f"--- Step {step['Step']} ---")
    print(f"AIC Values:")
    for feature, aic_value in step['AIC Values'].items():
        print(f"  {feature}: {aic_value:.4f}")
    print(f"Selected Feature: {step['Selected Feature']} with AIC: {step['Selected AIC']:.4f}")
    print(f"Remaining Predictors: {step['Remaining Predictors']}")
    print(f"Selected Predictors: {step['Selected Predictors']}\n")

# Final selected features
print("\n------Final Selected Features ---\n")
print(selected_features_aic)

print("\n...........................................................................\n")
print("\nbackward selection with AIC\n")

from statsmodels.api import OLS, add_constant


# Backward Selection based on AIC
def backward_selection_with_aic(X, y, significance_level=0.05):
    initial_features = list(X.columns)
    selected_features = list(initial_features)
    step_details = []

    while selected_features:
        # Fit model with all current features
        model = OLS(y, add_constant(X[selected_features])).fit()

        # Evaluate p-values for all remaining features
        p_values = model.pvalues.iloc[1:]  # Exclude intercept
        aic = model.aic

        max_p_value = p_values.max()
        worst_feature = p_values.idxmax()

        # Remove the feature with the highest p-value above significance level
        if max_p_value > significance_level:
            selected_features.remove(worst_feature)
            step_details.append({
                "Step": len(step_details) + 1,
                "AIC": aic,
                "Removed Feature": worst_feature,
                "Remaining Predictors": list(selected_features),
            })
        else:
            break

    return selected_features, step_details


# Perform backward selection based on AIC
selected_features_backward, steps_backward = backward_selection_with_aic(X, y)

# Print the details for each step
for step in steps_backward:
    print(f"--- Step {step['Step']} ---")
    print(f"AIC: {step['AIC']:.4f}")
    print(f"Removed Feature: {step['Removed Feature']}")
    print(f"Remaining Predictors: {step['Remaining Predictors']}\n")

# Final selected features
print("------------------Final Selected Features --------------------")
print("\nselected_features_backward\n",selected_features_backward)

# ,,,,,,,,,,,,,,,,,,,,,,,,,,............................................................................
import matplotlib.pyplot as plt

predictors_Model = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'maxchan']
response_Model = 'perfo'
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

remaining_data_Model = remaining_data[predictors_Model].copy()  # Ensure no SettingWithCopyWarning
remaining_data_Model['perfo'] = remaining_data['perfo']

# assumption 1a residual vs fitted after
# Step 2: Clean data (remove NaN or infinite values caused by transformations)
transformed_data_clean_Model = remaining_data_Model.replace([np.inf, -np.inf], np.nan).dropna()

X_Model = transformed_data_clean_Model[predictors_Model]
X_with_constant_Model = sm.add_constant(X_Model)

# Fit the model using the already transformed and cleaned data
y_transformed_Model = transformed_data_clean_Model['perfo']
X_transformed_Model = transformed_data_clean_Model[predictors_Model]
X_transformed_with_const_Model = sm.add_constant(X_transformed_Model)

model_Model = sm.OLS(y_transformed_Model, X_transformed_with_const_Model).fit()

# Get fitted values and residuals
fitted_values_Model = model_Model.fittedvalues
residuals_Model = model_Model.resid

# Plot Residuals vs Fitted values
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values_Model, residuals_Model, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')

plt.title("Residuals vs Fitted Values (Transformed Data)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

model_transformed_Model = sm.OLS(y_transformed_Model, X_transformed_with_const_Model).fit()
# Get fitted values and residuals
fitted_values_transformed_Model = model_transformed_Model.fittedvalues
residuals_transformed_Model = model_transformed_Model.resid

# single last
# Fit a quadratic polynomial (U-shaped curve) to the residuals
coeffs_Model = np.polyfit(fitted_values_transformed_Model, residuals_transformed_Model, deg=2)
poly_curve_Model = np.poly1d(coeffs_Model)

# Generate data points for the fitted curve
x_curve_Model = np.linspace(min(fitted_values_transformed_Model), max(fitted_values_transformed_Model), 200)
y_curve_Model = poly_curve_Model(x_curve_Model)

# Plot Residuals vs. Fitted Values with U-shaped curve (quadratic)
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values_transformed_Model, residuals_transformed_Model, alpha=0.7, label="Residuals")
plt.plot(x_curve_Model, y_curve_Model, color='blue', label="Quadratic Curve (U-shaped)", linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")
plt.title("Residuals vs Fitted Values with U-Shaped Quadratic Curve")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.show()

# individual

for predictor_Model in predictors_Model:
    # Extract the predictor's values and residuals
    predictor_values_Model = transformed_data_clean_Model[predictor_Model]
    residuals_transformed_Model = model_transformed_Model.resid

    # Fit a quadratic polynomial (U-shaped curve) to the residuals
    coeffs_Model = np.polyfit(predictor_values_Model, residuals_transformed_Model, deg=2)
    poly_curve_Model = np.poly1d(coeffs_Model)

    # Generate data points for the fitted curve
    x_curve_Model = np.linspace(min(predictor_values_Model), max(predictor_values_Model), 200)
    y_curve_Model = poly_curve_Model(x_curve_Model)

    # Plot Residuals vs Predictor with U-shaped curve (quadratic)
    plt.figure(figsize=(10, 6))
    plt.scatter(predictor_values_Model, residuals_transformed_Model, alpha=0.7, label="Residuals")
    plt.plot(x_curve_Model, y_curve_Model, color='blue', label="Quadratic Curve (U-shaped)", linewidth=2)
    plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")
    plt.title(f"Residuals vs {predictor_Model} with U-Shaped Quadratic Curve")
    plt.xlabel(predictor_Model)
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(True)
    plt.show()

# Residuals vs Order plot
plt.figure(figsize=(10, 6))

# Order of residuals based on their index
residual_order_Model = range(len(residuals_transformed_Model))

plt.plot(residual_order_Model, residuals_transformed_Model, marker='o', linestyle='', alpha=0.7, label="Residuals")
plt.axhline(y=0, color='red', linestyle='--', label="Zero Line")

plt.title("Residuals vs Order")
plt.xlabel("Order")
plt.ylabel("Residuals")
plt.grid(True)
plt.legend()
plt.show()
# .................................................


# Recompute the influence measures and Bonferroni p-values
influence_Model = model_transformed_Model.get_influence()

# Extract Cook's Distance
cooks_d_Model, p_values_Model = influence_Model.cooks_distance

# Thresholds for diagnostics
cooks_d_threshold_Model = 4 / len(y_transformed_Model)  # Typical Cook's Distance threshold
std_residuals_threshold_Model = 2  # Common threshold for standardized residuals
leverage_threshold_Model = 2 * (
            X_transformed_with_const_Model.shape[1] / len(y_transformed_Model))  # Leverage threshold

# Re-plot diagnostic plots with corrected Bonferroni p-values
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cook's Distance
sns.scatterplot(x=np.arange(len(cooks_d_Model)), y=cooks_d_Model, ax=axes1[0, 0], color='blue')
axes1[0, 0].axhline(y=cooks_d_threshold_Model, color='red', linestyle='--',
                    label=f"Threshold ({cooks_d_threshold_Model:.3f})")
axes1[0, 0].set_title("Cook's Distance")
axes1[0, 0].set_xlabel('Index')
axes1[0, 0].set_ylabel("Cook's Distance")
axes1[0, 0].grid(True)
axes1[0, 0].legend()

# Plot 2: Standardized Residuals
standardized_residuals_Model = influence_Model.resid_studentized_internal
sns.scatterplot(x=np.arange(len(standardized_residuals_Model)), y=standardized_residuals_Model, ax=axes1[0, 1],
                color='green')
axes1[0, 1].axhline(y=std_residuals_threshold_Model, color='red', linestyle='--',
                    label=f"Upper Threshold ({std_residuals_threshold_Model})")
axes1[0, 1].axhline(y=-std_residuals_threshold_Model, color='red', linestyle='--',
                    label=f"Lower Threshold (-{std_residuals_threshold_Model})")
axes1[0, 1].set_title('Standardized Residuals')
axes1[0, 1].set_xlabel('Index')
axes1[0, 1].set_ylabel('Standardized Residuals')
axes1[0, 1].grid(True)
axes1[0, 1].legend(loc='lower right')

# Plot 3: Hat Values (Leverage)
hat_values_Model = influence_Model.hat_matrix_diag
sns.scatterplot(x=np.arange(len(hat_values_Model)), y=hat_values_Model, ax=axes1[1, 0], color='red')
axes1[1, 0].axhline(y=leverage_threshold_Model, color='blue', linestyle='--',
                    label=f"Threshold ({leverage_threshold_Model:.3f})")
axes1[1, 0].set_title('Hat Values (Leverage)')
axes1[1, 0].set_xlabel('Index')
axes1[1, 0].set_ylabel('Hat Values')
axes1[1, 0].grid(True)
axes1[1, 0].legend()

# Correct Bonferroni p-values (for multiple comparisons)
bonferroni_p_values_Model = p_values_Model

# Plot 4: Bonferroni p-values
sns.scatterplot(x=np.arange(len(bonferroni_p_values_Model)), y=bonferroni_p_values_Model, ax=axes1[1, 1],
                color='purple')

axes1[1, 1].set_title('Bonferroni p-values')
axes1[1, 1].set_xlabel('Index')
axes1[1, 1].set_ylabel('Bonferroni p-value')
axes1[1, 1].grid(True)
plt.tight_layout()
plt.show()

# QQ plot # Create a final QQ plot for the transformed response variable 'perfo'

import scipy.stats as stats
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
stats.probplot(remaining_data_Model['perfo'], dist="norm", plot=plt)
plt.title("QQ Plot for Transformed Response Variable (perfo)")
plt.grid(True)
plt.show()

# Plot a histogram for the transformed response variable 'perfo'
plt.figure(figsize=(8, 6))
plt.hist(remaining_data_Model['perfo'], bins=15, edgecolor='black', alpha=0.7)
plt.title("Histogram of Transformed Response Variable (perfo)")
plt.xlabel("Transformed perfo")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)
plt.show()

# calculate the VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Subset the data to include only the selected predictors

selected_predictors_aic = remaining_data[predictors_Model].copy()
selected_X = selected_predictors_aic[predictors_Model]

# Add a constant column for VIF calculation
selected_X_with_const = sm.add_constant(selected_X)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Predictor'] = selected_X_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(selected_X_with_const.values, i)
                   for i in range(selected_X_with_const.shape[1])]

# Display the VIF values
print("\nVIF\n",vif_data)
