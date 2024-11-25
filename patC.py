import pandas as pd

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

print(df)
# Prepare the predictors and response variables
predictors = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']
response = 'perfo'
X = df[predictors]
y = df[response]
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
    print(f"Step {step['Step']}:")
    print(f"P-Values: {step['P-Values']}")
    print(f"Selected Feature: {step['Selected Feature']} with P-Value: {step['Selected P-Value']}")
    print(f"Remaining Predictors: {step['Remaining Predictors']}")
    print(f"Selected Predictors: {step['Selected Predictors']}\n")

# Print final selected features
print("Final Selected Features:")
print(selected_features_marginality)


print("..........................\n")
print("backward elimination with ftest")

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

print(formatted_backward_output)
print(formatted_final_output)

print("...........................................")
print("forward selection with AIC ")


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
print("--- Final Selected Features ---")
print(selected_features_aic)


print("..................................")
print("backward selection with AIC ")

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
print("--- Final Selected Features ---")
print(selected_features_backward)
