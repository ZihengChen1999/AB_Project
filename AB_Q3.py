import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import minimize
import os

###################################################################################################################################################################
# Question 3 Baseline Model


industry_returns = pd.read_csv('Industry_returns.csv', index_col=0, parse_dates=True)
factor_premiums = pd.read_csv('Factor_premiums.csv', index_col=0, parse_dates=True)

if not os.path.exists('Portflio_Construction'):
    os.makedirs('Portflio_Construction')

industry_returns = industry_returns / 100
factor_premiums = factor_premiums 

# Baseline minimum variance portfolio, allocating equal weight to each industry and calculate the return, variance, Sharpe ratio of the portfolio
baseline_weights = np.ones(12) / 12
baseline_monthly_return = np.dot(industry_returns, baseline_weights)
baseline_monthly_std = np.std(baseline_monthly_return)
baseline_annualized_return = np.mean(baseline_monthly_return) * 12
baseline_annualized_std = baseline_monthly_std * np.sqrt(12)
baseline_sharpe_ratio = baseline_annualized_return / baseline_annualized_std
baseline_cumulative_monthly_return = np.cumprod(1 + baseline_monthly_return) - 1

# Turn the monthly return into a dataframe and set the index to be the same as the industry returns
baseline_cumulative_monthly_return = pd.DataFrame(baseline_cumulative_monthly_return)
baseline_cumulative_monthly_return.index = industry_returns.index

# Draw the cumulative return plot, annote the return, std, and Sharpe ratio and save as jpg files
plt.plot(baseline_cumulative_monthly_return)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Baseline Minimum Variance Portfolio')
plt.annotate('Annualized Return: ' + str(round(baseline_annualized_return, 4)), xy=(0.05, 0.9), xycoords='axes fraction')
plt.annotate('Annualized Standard Deviation: ' + str(round(baseline_annualized_std, 4)), xy=(0.05, 0.85), xycoords='axes fraction')
plt.annotate('Sharpe Ratio: ' + str(round(baseline_sharpe_ratio, 4)), xy=(0.05, 0.8), xycoords='axes fraction')
plt.savefig(os.path.join('Portflio_Construction', 'Baseline_Minimum_Variance_Portfolio.jpg'))
plt.close()

###################################################################################################################################################################
# Question 3 Hierarchial Clustering

# Although there is some suspect of data snopping, we still use  the hierarchial clustering now to do the allocation, since actually the industry relationships is relative stable

# According to the hierarchial clustering, we can see that the 12 industries can be divided into 5 clusters, which are: 1. Utils 2. Energy 3. BusEq and Telcm 4. Hlth 5. DurBL, NoDur, Shope, Money, Chems, Nanuf, Other
# Then we give same weight to each cluster and then eually weight the industries in each cluster

# Build a dataframe to store the weight for each industry
industry_weights = pd.DataFrame(index=industry_returns.columns, columns=['Weight'])

# Mnually assign the weight for each cluster
industry_weights.loc[['Utils'], 'Weight'] = 0.2
industry_weights.loc[['Enrgy'], 'Weight'] = 0.2
industry_weights.loc[['BusEq', 'Telcm'], 'Weight'] = 0.2 / 2 
industry_weights.loc[['Hlth'], 'Weight'] = 0.2
industry_weights.loc[['Durbl', 'NoDur', 'Shops', 'Money', 'Chems', 'Manuf', 'Other'], 'Weight'] = 0.2 / 7

# Take out the weight for each industry as a numpy array
industry_weights = industry_weights['Weight'].values

# Do the same as basline models
cluster_monthly_return = np.dot(industry_returns, industry_weights)
cluster_monthly_std = np.std(cluster_monthly_return)
cluster_annualized_return = np.mean(cluster_monthly_return) * 12
cluster_annualized_std = cluster_monthly_std * np.sqrt(12)
cluster_sharpe_ratio = cluster_annualized_return / cluster_annualized_std
cluster_cumulative_monthly_return = np.cumprod(1 + cluster_monthly_return) - 1


# Turn the monthly return into a dataframe and set the index to be the same as the industry returns
cluster_cumulative_monthly_return = pd.DataFrame(cluster_cumulative_monthly_return)
cluster_cumulative_monthly_return.index = industry_returns.index


# Draw the cumulative return plot, annote the return, std, and Sharpe ratio and save as jpg files
plt.plot(cluster_cumulative_monthly_return)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Hierarchial Cluster Minimum Variance Portfolio')
plt.annotate('Annualized Return: ' + str(round(cluster_annualized_return, 4)), xy=(0.05, 0.9), xycoords='axes fraction')
plt.annotate('Annualized Standard Deviation: ' + str(round(cluster_annualized_std, 4)), xy=(0.05, 0.85), xycoords='axes fraction')
plt.annotate('Sharpe Ratio: ' + str(round(cluster_sharpe_ratio, 4)), xy=(0.05, 0.8), xycoords='axes fraction')
plt.savefig(os.path.join('Portflio_Construction', 'Cluster_Minimum_Variance_Portfolio.jpg'))
plt.close()



###################################################################################################################################################################
# Question 3 Structure Factor Model


# do not include the last row in factor_premiums, since it is the market return
factor_premiums = factor_premiums.iloc[:, :-1]

# calculate the covariance matrix of the factor premiums rolling window of 60 months
factor_cov = factor_premiums.rolling(60).cov()
# Svae the covariance matrix as a csv file
factor_cov.to_csv(os.path.join('Portflio_Construction', 'factor_cov.csv'))

# Reshape this into a 3D array 438 * 6 * 6
factor_cov = factor_cov.values.reshape(438, 6, 6)

# Calculate Q = (T / N)
Q = 60 / 6
# Calculate the maximum random eigenvalue (lambda_plus)
lambda_plus = (1 + np.sqrt(Q)) ** 2

# for each cov matrix in factor_cov, calculate the eigenvalues and eigenvectors 
# if the eigenvalue is smaller than lambda_plus, then replace it with 0
# Use a new array to store the new cov matrix called cleaned_factor_cov
cleaned_factor_cov = np.zeros((438, 6, 6))

for i in range(factor_cov.shape[0]):
    # if this one do not consist of nan
    if np.isnan(factor_cov[i]).any() == False:
        eigenvalues, eigenvectors = np.linalg.eig(factor_cov[i])
        for j in range(len(eigenvalues)):
            if eigenvalues[j] < lambda_plus:
                eigenvalues[j] = 0
        cleaned_factor_cov[i] = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))


# Shrinkage non-diagonal elements to 0 and save in a new array called diagonal_cleaned_factor_cov
diagonal_cleaned_factor_cov = np.zeros((438, 6, 6))
for i in range(cleaned_factor_cov.shape[0]):
    diagonal_cleaned_factor_cov[i] = np.diagflat(np.diag(cleaned_factor_cov[i]))

diagonal = 0.5
cleaned_factor_cov = diagonal * diagonal_cleaned_factor_cov + (1 - diagonal) * cleaned_factor_cov

# Shrinkage paramter shrinkage * c * I + (1 - shrinkage) * cleaned_factor_cov
shrinkage = 0.75
c = 3
shrinkaged_cleaned_factor_cov = shrinkage * c * np.identity(6) + (1 - shrinkage) * cleaned_factor_cov

# Save the shrinkaged_cleaned_factor_cov as a csv file


# I need to read in the factor loadings for each industry and stack factor loadings from different industry together at the same time and then use this to calculate the caovairnce matrix and then do the portfolio construction
# Read in the factor loadings for each industry and save them into different dataframes
industry_exposures = {}
for industry in industry_returns.columns:
    industry_exposures[industry] = pd.read_csv(os.path.join('Rolling_exposures', f'Reg_five_factor_model_{industry}.csv'), index_col=0).iloc[:,:-1]

# Stack the factor loadings from different industries together
stacked_exposures = pd.concat(industry_exposures.values(), axis=1)
# stacked_exposures.columns = industry_exposures.keys()

# For each row in the stacked_exposures reshape it into a 6 * 12 matrix
# Then multiply it with the shrinkaged_cleaned_factor_cov to get the covariance matrix for each industry
# Use a new array to store the covariance matrix for each time period
industry_cov = np.zeros((stacked_exposures.shape[0], 12, 12))
for i in range(stacked_exposures.shape[0]):
    exposure_matrix = stacked_exposures.iloc[i].values.reshape(12, 6)
    industry_cov[i] = np.dot(np.dot(exposure_matrix , shrinkaged_cleaned_factor_cov[i].reshape(6, 6)), exposure_matrix .T)
    
# Save the industry_cov as a csv file
industry_cov = pd.DataFrame(industry_cov.reshape(stacked_exposures.shape[0], 144))
industry_cov.index = stacked_exposures.index
industry_cov.to_csv(os.path.join('Portflio_Construction', 'industry_structured_cov.csv'))


def minimize_variance(cov_matrix):
    # Define the objective function (variance)
    def portfolio_variance(weights, cov_matrix):
        return weights @ cov_matrix @ weights

    # Define the constraints
    def portfolio_weights_sum_to_1(weights):
        return np.sum(weights) - 1


    # The constraints are passed to the minimizer as a dictionary
    constraints = ({'type': 'eq', 'fun': portfolio_weights_sum_to_1})

    # The bounds ensure the weights are between 0 and 1 (non-negative and sum to 1)
    bounds = tuple((0.0, 0.3) for _ in range(len(cov_matrix)))

    # Initial guess for the weights (equal distribution)
    init_guess = np.repeat(1 / len(cov_matrix), len(cov_matrix))

    # Use the SLSQP method to minimize the objective function with the constraints
    result = minimize(portfolio_variance, init_guess, 
                      args=(cov_matrix), method='SLSQP', 
                      bounds=bounds, constraints=constraints)

    # The optimal weights that minimize the portfolio variance
    optimal_weights = result.x

    return optimal_weights


# for each row in the industry_cov, calculate the portfolio weights sum to 1, non-negative, and minimize the variance
# Use a new array to store the portfolio weights for each time period
industry_portfolio_weights = np.zeros((industry_cov.shape[0], 12))
for i in range(industry_cov.shape[0]):
    industry_cov_matrix = industry_cov.iloc[i].values.reshape(12, 12)
    industry_portfolio_weights[i] = minimize_variance(industry_cov_matrix)

# Set index
industry_portfolio_weights = pd.DataFrame(industry_portfolio_weights)
industry_portfolio_weights.index = industry_cov.index
# Save the industry_portfolio_weights as a csv file
industry_portfolio_weights.to_csv(os.path.join('Portflio_Construction', 'industry_portfolio_weights.csv'))


# Take the last 437 rows of industtry_return
industry_returns = industry_returns.iloc[-377:,:]

structured_portfolio_monthly_return = industry_returns.values * industry_portfolio_weights.values

# Sum each row to get the monthly return for the structured portfolio
structured_portfolio_monthly_return = structured_portfolio_monthly_return.sum(axis=1)

# Annualize the returns
structured_portfolio_annual_return = structured_portfolio_monthly_return.mean() * 12
# Calculate the annualized volatility
structured_portfolio_annual_volatility = structured_portfolio_monthly_return.std() * np.sqrt(12)
# Calculate the annualized sharpe ratio
structured_portfolio_annual_sharpe_ratio = structured_portfolio_annual_return / structured_portfolio_annual_volatility
# Draw the cumulative returns
structured_portfolio_cumulative_return = (1 + structured_portfolio_monthly_return).cumprod() - 1

# Turn this into a dataframe and set index
structured_portfolio_cumulative_return = pd.DataFrame(structured_portfolio_cumulative_return)
structured_portfolio_cumulative_return.index = industry_returns.index

# Plot the cumulative returns and annote the sharpe ratio, volatility and return using exactly the same method as we did for the benchmark
plt.plot(structured_portfolio_cumulative_return)
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Structured Minimum Variance Portfolio')
plt.annotate('Annualized Return: ' + str(round(structured_portfolio_annual_return, 4)), xy=(0.05, 0.9), xycoords='axes fraction')
plt.annotate('Annualized Standard Deviation: ' + str(round(structured_portfolio_annual_volatility, 4)), xy=(0.05, 0.85), xycoords='axes fraction')
plt.annotate('Sharpe Ratio: ' + str(round(structured_portfolio_annual_sharpe_ratio, 4)), xy=(0.05, 0.8), xycoords='axes fraction')
plt.savefig(os.path.join('Portflio_Construction', 'Structured_Minimum_Variance_Portfolio.jpg'))
plt.close()





    





