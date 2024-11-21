#this gives scatterplot matrix in  seperate graphs
#######
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

# Define the columns for scatterplots
columns_to_plot = [
 'McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem',
 'minchan', 'maxchan', 'MemRan', 'ChPerMem', 'perfo'
]

# Loop through all possible pairs of columns to create scatterplots
for i, x_col in enumerate(columns_to_plot):
  for y_col in columns_to_plot[i+1:]:
    plt.figure(figsize=(8, 6))
    sns.regplot(
    x=x_col,
    y=y_col,
    data=df1,
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'}
 )
    plt.title(f'Scatterplot of {x_col} vs {y_col}', fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

