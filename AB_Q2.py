import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import os

colors = ["green", "white", "red"]
cmap = LinearSegmentedColormap.from_list("", colors)

###################################################################################################################################################################
# Question 2

industry_returns = pd.read_csv('Industry_returns.csv', index_col=0, parse_dates=True)
factor_premiums = pd.read_csv('Factor_premiums.csv', index_col=0, parse_dates=True)

if not os.path.exists('Clustering'):
    os.makedirs('Clustering')
# Calculate the residuals of the adjusted five factor model Regularized regression do not have resid attribute, so we use the fittedvalues subtract the actual values to get the residuals
residuals = pd.DataFrame()
for i in range(12):
    industry = industry_returns.columns[i]
    y = industry_returns[industry]
    x = factor_premiums[['Mkt_minus_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
    model = sm.OLS(y, sm.add_constant(x)).fit_regularized(alpha = 0.5, L1_wt = 0.5)
    residuals[industry] = model.fittedvalues - y
residuals.columns = industry_returns.columns

# Calculate correlation matrix and plot the heatmap and save as jpg files
corr = residuals.corr().round(2)
sns.heatmap(corr, annot=True, cmap=cmap, vmax=1, center=0, vmin=-1, linewidths=0.2)
plt.savefig(os.path.join('Clustering', 'Residual_correlation.jpg'))
plt.close()

# Calculate the covariance matrix and plot the heatmap and save as jpg files
cov = residuals.cov().round(2)
sns.heatmap(cov, annot=True, cmap=cmap, linewidths=0.2)
plt.savefig(os.path.join('Clustering', 'Residual_covariance.jpg'))
plt.close()

corr = industry_returns.corr().round(2)
sns.heatmap(corr, annot=True, cmap=cmap, vmax=1, center=0.5, vmin=0, linewidths=0.2)
plt.savefig(os.path.join('Clustering', 'Industry_returns_correlation.jpg'))
plt.close()

# Calculate the covariance matrix and plot the heatmap and save as jpg files
cov = industry_returns.cov().round(2)
sns.heatmap(cov, annot=True, cmap=cmap, linewidths=0.2)
plt.savefig(os.path.join('Clustering', 'Industry_returns_covariance.jpg'))
plt.close()

# Perform hierarchical clustering on industry returns
# Calculate the distance matrix
distance = pdist(industry_returns.T, metric='correlation')
# Perform hierarchical clustering
linkage_matrix = linkage(distance, method='average', metric='correlation')
# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=industry_returns.columns, leaf_rotation=90, leaf_font_size=10)
plt.savefig(os.path.join('Clustering', 'Industry_returns_dendrogram.jpg'))

# Seperate period into beofre 2002 and after 2002
industry_returns_before = industry_returns.loc[:'2002-12-31']
industry_returns_after = industry_returns.loc['2003-01-01':]

# Perform hierarchical clustering on industry returns before 2002
# Calculate the distance matrix
distance = pdist(industry_returns_before.T, metric='correlation')
# Perform hierarchical clustering
linkage_matrix = linkage(distance, method='average', metric='correlation')
# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=industry_returns_before.columns, leaf_rotation=90, leaf_font_size=10)
# Add titel
plt.title('Hierarchical Clustering Dendrogram (Before 2002)')
plt.savefig(os.path.join('Clustering', 'Industry_returns_before_dendrogram.jpg'))
plt.close()

# Perform hierarchical clustering on industry returns after 2002
# Calculate the distance matrix
distance = pdist(industry_returns_after.T, metric='correlation')
# Perform hierarchical clustering
linkage_matrix = linkage(distance, method='average', metric='correlation')
# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=industry_returns_after.columns, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram (After 2002)')
plt.savefig(os.path.join('Clustering', 'Industry_returns_after_dendrogram.jpg'))
