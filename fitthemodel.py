import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import probplot

#this gives scatterplot matrix in single plot

import pandas as pd

# Load the uploaded Excel file
file_path = '/Users/brahmendrajayaraju/Desktop/Book1.xlsx'
data = pd.ExcelFile(file_path)

# Extract sheet names to see what's available
sheet_names = data.sheet_names
print("Sheet names:", sheet_names)

# Load data from the first sheet (Sheet1) into a DataFrame
df1 = data.parse(sheet_names[0])

# Add new columns: Memory Range and Channels per Memory
#df1['ChanEff'] = df1['maxchan'] - df1['minchan']
df1['CacheCycleRatio'] = df1['cachemem'] / df1['McycTime']


# Prepare the data
predictors = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan','CacheCycleRatio']
X = df1[predictors]
y = df1['perfo']

# Add a constant to the predictors for the intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Summary of the regression model
summary = model.summary()

# Residuals
residuals = model.resid
fitted_values = model.fittedvalues

# Create diagnostic plots
def diagnostic_plots():
    # Residuals vs Fitted values
    plt.figure(figsize=(18, 10))
    plt.scatter(fitted_values, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals vs Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Residuals vs Order (to check independence)
    plt.figure(figsize=(18, 10))
    plt.plot(residuals, marker='o', linestyle='none')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals vs Order')
    plt.xlabel('Order')
    plt.ylabel('Residuals')
    plt.show()

    # Histogram of residuals for normality check
    plt.figure(figsize=(18, 10))
    plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

    # Q-Q plot for normality check
    plt.figure(figsize=(18, 10))
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()

    # Cook's distance plot for influential points
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    plt.figure(figsize=(18, 10))
    plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",", basefmt=" ")
    plt.axhline(4 / len(cooks_d), color='red', linestyle='--', linewidth=1, label='Threshold (4/n)')
    plt.title("Cook's Distance")
    plt.xlabel('Observation Index')
    plt.ylabel("Cook's Distance")
    plt.legend()
    plt.show()

# Display diagnostic plots


print(summary)

# Display the regression summary with p-values in exponential (scientific) notation
summary_as_text = model.summary().as_text()

# Extract p-values from the model results
p_values = model.pvalues

# Display p-values in exponential format
p_values_exp = p_values.apply(lambda x: f"{x:.2e}")

# Create a clean DataFrame to display coefficients and p-values
coeff_summary = pd.DataFrame({
    "Coefficient": model.params,
    "P-value (Exponential)": p_values_exp
})
print(coeff_summary)

diagnostic_plots()

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF data
print(vif_data)

# Add a constant to the predictors for the intercept
X = sm.add_constant(X)

# Calculate VIF for each predictor
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Iteratively remove high-VIF predictors
def remove_high_vif(X, threshold=5.0):
    while True:
        vif = calculate_vif(X)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            max_vif_idx = vif["VIF"].idxmax()
            dropped_var = vif.loc[max_vif_idx, "Variable"]
            print(f"Dropping '{dropped_var}' with VIF={max_vif}")
            X = X.drop(columns=[dropped_var])
        else:
            break
    return X

# Initial VIF calculation
print("Initial VIF:")
vif_initial = calculate_vif(X)
print(vif_initial)

# Feature Engineering: Add 'MaiMemRange' and drop 'minMaiMem' and 'maxMaiMem'
df1['MaiMemRange'] = df1['maxMaiMem'] - df1['minMaiMem']
predictors = ['McycTime', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio', 'MaiMemRange']
X = df1[predictors]
X = sm.add_constant(X)

# Iteratively remove high-VIF predictors
X_reduced = remove_high_vif(X)

# Final VIF calculation
print("Final VIF after reduction:")
vif_final = calculate_vif(X_reduced)
print(vif_final)

# Fit the regression model
model = sm.OLS(y, X_reduced).fit()

# Display the summary of the model
print(model.summary())

# Optional: PCA for further dimensionality reduction
pca = PCA(n_components=min(len(X_reduced.columns), len(X)))
X_pca = pca.fit_transform(X_reduced)

print("PCA Explained Variance Ratios:")
print(pca.explained_variance_ratio_)