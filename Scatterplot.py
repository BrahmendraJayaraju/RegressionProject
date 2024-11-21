#this gives scatterplot matrix in single plot
#brahmendra
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
df1['MemRan'] = df1['maxMaiMem'] - df1['minMaiMem']
df1['ChPerMem'] = df1['maxchan'] / df1['maxMaiMem']

# Reorder columns to place the new columns before the 'performance' column
columns = list(df1.columns)
performance_index = columns.index('perfo')

columns.remove('perfo') # Remove 'performance' from its current position
columns.insert(columns.index('ChPerMem') + 1, 'perfo')

# Reassign the reordered columns to the DataFrame
df1 = df1[columns]


# Display all rows and columns
pd.set_option('display.max_rows', None) # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', 1000) # Set the display width to fit data

# Print all rows of the updated DataFrame
print(df1)


import seaborn as sns
import matplotlib.pyplot as plt

# Define the columns to include in the plot
columns_to_plot = [
 'McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem',
 'minchan', 'maxchan', 'MemRan', 'ChPerMem', 'perfo'
]

# Create a pairplot with regression lines
pairplot = sns.pairplot(
   df1[columns_to_plot],
   kind="reg", # Adds regression lines
   diag_kind="kde", # Kernel density estimation for diagonal
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