from patC import remaining_data_Model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Assuming `remaining_data_Model` contains the dataset
# Load the data
data = remaining_data_Model
# Define the response and predictors
response = 'perfo'
predictors = ['McycTime', 'minMaiMem', 'maxMaiMem', 'cachemem', 'maxchan']
# Construct the formula for regression
formula = f"{response} ~ {' + '.join(predictors)}"

# Fit the model
model = smf.ols(formula=formula, data=remaining_data_Model).fit()

# Obtain parameter estimates
params = model.params
# Perform type III ANOVA
anova_results = sm.stats.anova_lm(model, typ=3)
print("\nsummary of model\n", model.summary())
# Extract standard errors and p-values from the model
standard_errors = model.bse  # Standard errors
p_values = model.pvalues  # p-values
print("\nP value seperately\n", p_values)
print("\nSD seperately\n", standard_errors)

# Calculate R-squared
r_squared = model.rsquared

# Determine the number of rows and columns for the grid
rows, cols = 3, 2  # 3 rows and 2 columns

# Create subplots for the effect plots
fig, axes = plt.subplots(rows, cols, figsize=(12, 12), constrained_layout=True)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot each predictor
for i, predictor in enumerate(predictors):
    sm.graphics.plot_ccpr(model, predictor, ax=axes[i])
    axes[i].set_title(f"Effect Plot for {predictor}")
    axes[i].set_xlabel(predictor)
    axes[i].set_ylabel(response)

# Turn off unused subplots
for j in range(len(predictors), len(axes)):
    axes[j].axis('off')

plt.suptitle("\nEffect Plots for Predictors\n", fontsize=16)
plt.show()

# Display ANOVA table and parameter estimates
print("\nAnova Table\n",anova_results)
print("\nParameter Estimates:\n")
print(params)

# Summary for R-squared and interpretation
interpretation = {
    "R_squared": r_squared,
    "Interpretation": "The R-squared value represents the proportion of variance in the response variable 'perfo' "
                      "that is explained by the predictors in the model."
}
print("\nR^2\n",interpretation)




