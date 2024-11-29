from patC import remaining_data_Model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Assuming `remaining_data_Model` contains the dataset
# Load the data
data = remaining_data_Model
# Define the response and predictors
response = 'Inv_Sqrt_perfo'
predictors = ['Log_McycTime', 'Log_minMaiMem', 'Log_maxMaiMem', 'Log_cachemem', 'Log_maxchan']
# Construct the formula for regression

remaining_data_Model = remaining_data_Model.rename(columns={
    "Log(McycTime)": "Log_McycTime",
    "Log(minMaiMem)": "Log_minMaiMem",
    "Log(maxMaiMem)": "Log_maxMaiMem",
    "Log(cachemem)": "Log_cachemem",
    "Log(maxchan)": "Log_maxchan",
    "Inv(Sqrt(perfo))": "Inv_Sqrt_perfo"
})

formula =f"{response} ~ {' + '.join(predictors)}"

print(remaining_data_Model)
print(formula)
# Fit the model
model = smf.ols(formula=formula, data=remaining_data_Model).fit()

# Obtain parameter estimates
params = model.params
# Perform type III ANOVA
anova_results = sm.stats.anova_lm(model, typ=3)
print("\nsummary of model -Model 1\n", model.summary())
# Extract standard errors and p-values from the model
standard_errors = model.bse  # Standard errors
p_values = model.pvalues  # p-values
print("\nP value seperately -Model 1\n", p_values)
print("\nSE seperately- Model 1\n", standard_errors)

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
    axes[i].set_title(f"Effect Plot for -Model 1 {predictor}",fontsize=16, fontweight='bold')
    axes[i].set_xlabel(predictor,fontsize=16)
    axes[i].set_ylabel(response,fontsize=16)

# Turn off unused subplots
for j in range(len(predictors), len(axes)):
    axes[j].axis('off')

plt.suptitle("\nEffect Plots for Predictors - Model 1\n", fontsize=16,fontweight='bold')
plt.show()


# Display ANOVA table and parameter estimates
print("\nAnova Table -Model 1\n",anova_results)
print("\nParameter Estimates -Model 1:\n")
print(params)

# Summary for R-squared and interpretation
interpretation = {
    "R_squared": r_squared,
    "Interpretation": "The R-squared value represents the proportion of variance in the response variable 'perfo' "
                      "that is explained by the predictors in the model."
}
print("\nR^2 -Model 1 \n",interpretation)