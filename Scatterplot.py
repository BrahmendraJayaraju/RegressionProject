import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix  # Import scatter_matrix function
import seaborn as sns
import matplotlib.pyplot as plt

# Load the uploaded Excel file
file_path = '/Users/brahmendrajayaraju/Desktop/Book1.xlsx'
data = pd.ExcelFile(file_path)

# Extract sheet names to see what's available
sheet_names = data.sheet_names
print("Sheet names:", sheet_names)

# Load data from the first sheet (Sheet1) into a DataFrame
df1 = data.parse(sheet_names[0])

# Add new columns: Memory Range and Channels per Memory
df1['CacheCycleRatio'] = df1['cachemem'] / df1['McycTime']

# Reorder columns to place the new columns before the 'performance' column
columns = list(df1.columns)
performance_index = columns.index('perfo')

columns.remove('perfo')  # Remove 'performance' from its current position
columns.insert(columns.index('CacheCycleRatio') + 1, 'perfo')

# Reassign the reordered columns to the DataFrame
df1 = df1[columns]

# Display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set the display width to fit data

# Print all rows of the updated DataFrame
print(df1)

# Define the columns to include in the plot
columns_to_plot = [
    'McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem',
    'minchan', 'maxchan', 'CacheCycleRatio', 'perfo'
]

# Create a pairplot with regression lines
pairplot = sns.pairplot(
    df1[columns_to_plot],
    kind="reg",  # Adds regression lines
    diag_kind="kde",  # Kernel density estimation for diagonal
    plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.6}}
)

# Adjust the plot layout to fit labels
pairplot.fig.subplots_adjust(top=0.95, bottom=0.2, left=0.05, right=0.95)

# Set the title for the figure
pairplot.fig.suptitle('Scatter Plot Matrix with Regression Lines', y=1.02, fontsize=16)

# Rotate x-axis labels for better visibility
for ax in pairplot.axes.flat:
    if ax.xaxis.get_label():
        ax.xaxis.set_tick_params(rotation=45)
plt.show()

# Apply log transformation to relevant columns
log_transformed_data = df1.copy()
columns_to_log_transform = ['minMaiMem', 'maxMaiMem', 'cachemem', 'CacheCycleRatio', 'perfo']

for column in columns_to_log_transform:
    log_transformed_data[f'log_{column}'] = np.log1p(log_transformed_data[column])

# Define the columns to include in the scatterplot matrix
log_columns_to_plot = [f'log_{col}' for col in columns_to_log_transform] + ['McycTime', 'minchan', 'maxchan']

# Create a pairplot with regression lines for log-transformed variables
pairplot = sns.pairplot(
    log_transformed_data[log_columns_to_plot],
    kind="reg",  # Adds regression lines
    diag_kind="kde",  # Kernel density estimation for diagonal plots
    plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.6}}
)

# Adjust the plot layout to fit labels
pairplot.fig.subplots_adjust(top=0.95, bottom=0.2, left=0.05, right=0.95)

# Set the title for the figure
pairplot.fig.suptitle('Scatter Plot Matrix with Log-Transformed Variables and Regression Lines', y=1.02, fontsize=16)

# Rotate x-axis labels for better visibility
for ax in pairplot.axes.flat:
    if ax.xaxis.get_label():
        ax.xaxis.set_tick_params(rotation=45)

# Show the plot
plt.show()


import statsmodels.api as sm

# Preparing data for regression (using log-transformed variables)
X = log_transformed_data[['log_minMaiMem', 'log_maxMaiMem', 'log_cachemem', 'log_CacheCycleRatio', 'McycTime', 'minchan', 'maxchan']]
X = sm.add_constant(X)  # Add a constant term for the regression
y = log_transformed_data['log_perfo']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Summary of the model
model_summary = model.summary()
print(model_summary)


# Extract p-values and format them in exponential form
p_values = model.pvalues.apply(lambda x: f"{x:.2e}")

# Create a DataFrame for better display
p_values_summary = pd.DataFrame({
    'Predictor': model.pvalues.index,
    'P-value (Exponential Form)': p_values
})

# Display the p-values in exponential form
print(p_values_summary)

# Residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# Residuals vs Fitted Plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

# Residuals vs Order Plot
plt.figure(figsize=(8, 6))
plt.plot(residuals, marker='o', linestyle='', alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Observation Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Order')
plt.show()

# Q-Q Plot for Normality
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# Cook's Distance (for influential points)
influence = model.get_influence()
cooks_d, _ = influence.cooks_distance

plt.figure(figsize=(8, 6))
plt.stem(cooks_d)
plt.axhline(4 / len(residuals), color='red', linestyle='--', linewidth=1)
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance Plot")
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['Predictor'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF values
print(vif_data)


# Perform hypothesis test for each predictor based on p-values
hypothesis_test_results = pd.DataFrame({
    'Predictor': model.pvalues.index,
    'P-value': model.pvalues,
    'Significant (α=0.05)': model.pvalues < 0.05
})

# Display hypothesis test results
print(hypothesis_test_results)
