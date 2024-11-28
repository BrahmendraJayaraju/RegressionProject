
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm



# Load the uploaded Excel file
file_path = '/Users/brahmendrajayaraju/Desktop/Book1.xlsx'
data = pd.read_excel(file_path)
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

# Select predictors and response variable
selected_columns = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio', 'perfo']
selected_data = data[selected_columns]

# Define predictors (X) and response (y)
X = data[['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'minchan', 'maxchan', 'CacheCycleRatio']]
y = data['perfo']

# Add a constant (intercept term)
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())

# Create a matrix plot (pairplot) with marker as 'x'
pairplot = sns.pairplot(
    selected_data,
    diag_kind='kde',
    kind="reg",
    plot_kws={'marker': 'x', 'scatter_kws': {'s': 10}},  # Marker as 'x' and adjust size
    diag_kws={'fill': True}  # Replace 'shade' with 'fill'
)

# Rotate x-axis labels after the pairplot is created
for ax in pairplot.axes.flatten():
    ax.set_xticks(ax.get_xticks())  # Ensure fixed tick positions
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust layout for better spacing
plt.suptitle("Matrix Plot of Predictors and Response Variable", y=1.05, fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Prevent title overlap
plt.show()

# Calculate the correlation matrix
correlation_matrix = selected_data.corr()
# Display the correlation matrix to the user
print("Correlation Matrix For Selected Variables\n")
print(correlation_matrix)
# Extract and analyze key relationships
response_correlation = correlation_matrix['perfo'][:-1]  # Exclude self-correlation
# Identify relationships among predictors
predictor_correlation = correlation_matrix.iloc[:-1, :-1]  # Exclude 'perfo'
# Summarize findings
findings = {
    "Response Relationships": response_correlation,
    "Predictor Relationships": predictor_correlation.describe(),
    "Extreme Values": selected_data.describe(percentiles=[0.01, 0.99]),
    "Potential Model Assumption Violations": {
        "Multicollinearity": predictor_correlation.max().max(),
        "Linearity Issues": response_correlation.abs().min()
    }
}




import numpy as np

# Boxplot for 'perfo' grouped by 'Manufacturer'
data.boxplot(column='perfo', by='Manufacturer', grid=False, rot=90,figsize=(18, 10))
plt.title('Boxplot of Performance by Manufacturer', fontsize=16, fontweight='bold')
plt.xlabel('Manufacturer',fontsize=14, fontweight='bold')
plt.ylabel('Performance (perfo)',fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# Convert Manufacturer to numeric codes and scale for more spacing
manufacturer_numeric = data['Manufacturer'].astype('category').cat.codes
scaling_factor = 2  # Adjust scaling factor to increase spacing
scaled_manufacturer_numeric = manufacturer_numeric * scaling_factor
# Set the figure size explicitly and BEFORE the plot commands
fig, ax = plt.subplots(figsize=(16, 10))  # Use subplots to ensure size is applied
# Create scatter plot with increased x-axis spacing
ax.scatter(scaled_manufacturer_numeric, data['perfo'], alpha=0.7, marker='x', c='purple', label='perfo')
# Adjust xticks to match the scaled positions
xtick_positions = np.arange(len(data['Manufacturer'].unique())) * scaling_factor
ax.set_xticks(ticks=xtick_positions)
ax.set_xticklabels(data['Manufacturer'].unique(), rotation=90)
# Customize plot appearance
ax.set_title('Scatter Plot of Performance by Manufacturer', fontsize=16, fontweight='bold')
ax.set_xlabel('Manufacturer', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance (perfo)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()  # Ensure proper spacing around the plot
plt.show()



# # Clean column names
# data.columns = data.columns.str.strip()
# # Handle the 'ModelName' column to ensure uniform data type
# data['ModelName'] = data['ModelName'].apply(lambda x: str(x) if not isinstance(x, str) else x)
# # Boxplot for 'perfo' grouped by 'ModelName'
# data.boxplot(column='perfo', by='ModelName', grid=False, rot=90)
# plt.title('Boxplot of Performance by ModelName')
# plt.suptitle('')  # Suppress automatic title
# plt.xlabel('ModelName')
# plt.ylabel('Performance (perfo)')
# plt.show()
#
#
#
#
# # Adding jitter to ModelName categories for better visualization
# modelname_numeric = data['ModelName'].astype('category').cat.codes
# jitter = 0.1 * (pd.Series(modelname_numeric).sample(frac=1).reset_index(drop=True) - 0.5)
# plt.scatter(modelname_numeric + jitter, data['perfo'], alpha=0.7, marker='x', c='purple', label='perfo')
# plt.xticks(ticks=range(len(data['ModelName'].unique())), labels=data['ModelName'].unique(), rotation=90)
# plt.title('Scatter Plot of Performance by ModelName')
# plt.xlabel('ModelName')
# plt.ylabel('Performance (perfo)')
# plt.grid(False)
# plt.legend()
# plt.show()


# Printing findings with values on the next line
# for key, value in findings.items():
#     if isinstance(value, dict):  # Handle nested dictionaries
#         print(f"{key}:")
#         for sub_key, sub_value in value.items():
#             print(f"  {sub_key}:\n    {sub_value}")
#     else:
#         print(f"{key}:\n{value}")