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

###################################################################################################################################################################
# Data cleaning and preprocessing

# read csv files
raw_five_factor_premium_file_path = os.path.join(os.getcwd(), 'Raw_Data', 'Raw_5_Factors.csv')
five_factor_premiums = pd.read_csv(raw_five_factor_premium_file_path, skiprows=range(0, 3), nrows=720, index_col=0)

raw_mom_premium_file_path = os.path.join(os.getcwd(), 'Raw_Data', 'Raw_Momentum.csv')
mom_premiums = pd.read_csv(raw_mom_premium_file_path, skiprows=range(0, 13), nrows=1158, index_col=0)
# Rename column name to MOM
mom_premiums.rename(columns={'Mom   ': 'MOM'}, inplace=True)

industry_returns_file_path = os.path.join(os.getcwd(), 'Raw_Data', 'Raw_12_Industries.csv')
industry_returns = pd.read_csv(industry_returns_file_path, skiprows=range(0, 11), nrows=1164, index_col=0)


industry_returns.index = pd.to_datetime(industry_returns.index, format='%Y%m') + pd.offsets.MonthEnd(1)
industry_returns = industry_returns.rename(columns={"Hlth ": "Hlth"})
industry_returns = industry_returns.loc['1987-01-01':]
industry_returns.to_csv('Industry_returns.csv')

# Combine five factor premiums and momentum premiums
factor_premiums = pd.concat([five_factor_premiums, mom_premiums], axis=1, join='inner')
factor_premiums = factor_premiums.rename(columns={"Mkt-RF": "Mkt_minus_RF"})
factor_premiums.index = pd.to_datetime(factor_premiums.index, format='%Y%m') + pd.offsets.MonthEnd(1)
factor_premiums = factor_premiums.loc['1987-01-01':]
factor_premiums = factor_premiums[[col for col in factor_premiums.columns if col != 'RF'] + ['RF']]
factor_premiums.to_csv('Factor_premiums.csv')

if not os.path.exists('Factor_premiums_graph'):
    os.makedirs('Factor_premiums_graph')

###################################################################################################################################################################
# Question 1.a


# Plot each factor premium one by one 
for i in range(7):
    plt.plot(factor_premiums.iloc[:, i].rolling(window=12).mean())
    plt.title(factor_premiums.columns[i])
    plt.savefig(os.path.join('Factor_premiums_graph', factor_premiums.columns[i] + '.jpg'))
    plt.close()
    sns.displot(factor_premiums.iloc[:, i], kde = True, height=5, aspect=2)
    plt.savefig(os.path.join('Factor_premiums_graph', factor_premiums.columns[i] + '_distribution.jpg'))
    plt.close()

# Calculate summary statistics and keep two decimal places
factor_premiums.describe().round(2).to_csv('Factor_premiums_summary.csv')
industry_returns.describe().round(2).to_csv('Industry_returns_summary.csv')
    
# Calculate the correlation matrix and output using seaborn heatmap, heavy color means high correlation
colors = ["green", "white", "red"]
cmap = LinearSegmentedColormap.from_list("", colors)
corr = factor_premiums.iloc[:, :-1].corr().round(2)
sns.heatmap(corr, annot=True, cmap=cmap, vmax=1, center=0, vmin=-1, linewidths=0.2)
plt.savefig(os.path.join('Factor_premiums_graph', 'Correlation.jpg'))
plt.close()

if not os.path.exists('Industry_returns_graph'):
    os.makedirs('Industry_returns_graph')

corr = industry_returns.iloc[:, :-1].corr().round(2) 
sns.heatmap(corr, annot=True, cmap=cmap, vmax=1, center=0.5, vmin=0, linewidths=0.2)
plt.savefig(os.path.join('Industry_returns_graph', 'Correlation.jpg'))
# High correlation between SMB and HML detected
plt.close()

OLS_Data = pd.concat([factor_premiums, industry_returns], axis=1, join='inner')
# Save the data to csv file
OLS_Data.to_csv('OLS_Data.csv')

# Three/Four factor model

if not os.path.exists('Three_factor_model'):
    os.makedirs('Three_factor_model')
for i in range(12):
    model = smf.ols(formula=f'{industry_returns.columns[i]} ~ Mkt_minus_RF + SMB + HML + MOM', data=OLS_Data).fit()
    with open(os.path.join('Three_factor_model', f'{industry_returns.columns[i]}_3_factor_model.txt'), 'w') as fh:
        fh.write(model.summary().as_text())

    if i == 0:
        three_factor_model = pd.DataFrame({'Industry': industry_returns.columns[i], 'Intercept': model.params[0], 'Mkt_minus_RF': model.params[1], 'SMB': model.params[2], 'HML': model.params[3], 'MOM':model.params[4], 'R^2': model.rsquared, 'Adj_R^2': model.rsquared_adj, 'Residual_mean': model.resid.mean(), 'Residual_std': model.resid.std() ** (1/2)}, index=[i])
    else:
        three_factor_model = pd.concat([three_factor_model, pd.DataFrame({'Industry': industry_returns.columns[i], 'Intercept': model.params[0], 'Mkt_minus_RF': model.params[1], 'SMB': model.params[2], 'HML': model.params[3], 'MOM':model.params[4], 'R^2': model.rsquared, 'Adj_R^2': model.rsquared_adj, 'Residual_mean': model.resid.mean(), 'Residual_std': model.resid.std() ** (1/2)}, index=[i])], axis=0)
   
    # Plot the residuals rolling smoothed and save as jpg files
    plt.plot(model.resid.rolling(window=60).std( ))
    plt.title(f'{industry_returns.columns[i]}_3_factor_model')
    plt.savefig(os.path.join('Three_factor_model', f'{industry_returns.columns[i]}_3_factor_model.jpg'))
    plt.close()

three_factor_model.to_csv('Three_factor_model.csv')

# Five/Six factor model
if not os.path.exists('Five_factor_model'):
    os.makedirs('Five_factor_model')
for i in range(12):
    model = smf.ols(formula=f'{industry_returns.columns[i]} ~ Mkt_minus_RF + SMB + HML + RMW + CMA + MOM', data=OLS_Data).fit()
    with open(os.path.join('Five_factor_model', f'{industry_returns.columns[i]}_5_factor_model.txt'), 'w') as fh:
        fh.write(model.summary().as_text())

    if i == 0:
        five_factor_model = pd.DataFrame({'Industry': industry_returns.columns[i], 'Intercept': model.params[0], 'Mkt_minus_RF': model.params[1], 'SMB': model.params[2], 'HML': model.params[3], 'RMW': model.params[4], 'CMA': model.params[5], 'MOM':model.params[6], 'R^2': model.rsquared, 'Adj_R^2': model.rsquared_adj,'Residual_mean': model.resid.mean(), 'Residual_std': model.resid.std() ** (1/2)}, index=[i])
    else:
        five_factor_model = pd.concat([five_factor_model, pd.DataFrame({'Industry': industry_returns.columns[i], 'Intercept': model.params[0], 'Mkt_minus_RF': model.params[1], 'SMB': model.params[2], 'HML': model.params[3], 'RMW': model.params[4], 'CMA': model.params[5], 'MOM':model.params[6], 'R^2': model.rsquared, 'Adj_R^2': model.rsquared_adj,'Residual_mean': model.resid.mean(), 'Residual_std': model.resid.std() ** (1/2)}, index=[i])], axis=0)
    
    # Plot the residuals and save as jpg files
    plt.plot(model.resid.rolling(window=60).std( ))
    plt.title(f'{industry_returns.columns[i]}_5_factor_model')
    plt.savefig(os.path.join('Five_factor_model', f'{industry_returns.columns[i]}_5_factor_model.jpg'))
    plt.close()

five_factor_model.to_csv('Five_factor_model.csv')

# We noticed that there are high corr between (SMB, RMW) and (HML, CMA), (HML, RMW), so we regress SMB on RMW and regress CMW AND RMW on HML and use the residuals as new factors
# Adjusted five/six fatcor model
if not os.path.exists('Adjusted_five_factor_model'):
    os.makedirs('Adjusted_five_factor_model')

# Regress SMB on RMW and add residuals to the OLS_Data as new factor "Adj_SMB"
model = smf.ols(formula='SMB ~ RMW', data=OLS_Data).fit()
OLS_Data['Adj_SMB'] = model.resid

# Regress CMA and RMW on HML and add residuals to the OLS_Data as new factor "Adj_CMA"
model = smf.ols(formula='HML ~ CMA + RMW', data=OLS_Data).fit()
OLS_Data['Adj_HML'] = model.resid


corr = OLS_Data[["Mkt_minus_RF", "Adj_SMB", "Adj_HML", "RMW", "CMA",  "MOM"]].corr().round(2)
sns.heatmap(corr, annot=True, cmap=cmap, vmax=1, center=0, vmin=-1, linewidths=0.2)
plt.savefig(os.path.join('Factor_premiums_graph', 'Adjusted_correlation.jpg'))
plt.close()

# Now do regression using new six factors
for i in range(12):
    model = smf.ols(formula=f'{industry_returns.columns[i]} ~ Mkt_minus_RF + Adj_SMB + Adj_HML + RMW + CMA + MOM', data=OLS_Data).fit()
    with open(os.path.join('Adjusted_five_factor_model', f'{industry_returns.columns[i]}_Adj_5_factor_model.txt'), 'w') as fh:
        fh.write(model.summary().as_text())

    if i == 0:
        adjusted_five_factor_model = pd.DataFrame({'Industry': industry_returns.columns[i], 'Intercept': model.params[0], 'Mkt_minus_RF': model.params[1], 'Adj_SMB': model.params[2], 'Adj_HML': model.params[3], 'RMW': model.params[4], 'CMA': model.params[5], 'MOM':model.params[6], 'R^2': model.rsquared, 'Adj_R^2': model.rsquared_adj, 'Residual_mean': model.resid.mean(), 'Residual_std': model.resid.std() ** (1/2)}, index=[i])
    else:
        adjusted_five_factor_model = pd.concat([adjusted_five_factor_model, pd.DataFrame({'Industry': industry_returns.columns[i], 'Intercept': model.params[0], 'Mkt_minus_RF': model.params[1], 'Adj_SMB': model.params[2], 'Adj_HML': model.params[3], 'RMW': model.params[4], 'CMA': model.params[5], 'MOM':model.params[6], 'R^2': model.rsquared, 'Adj_R^2': model.rsquared_adj, 'Residual_mean': model.resid.mean(), 'Residual_std': model.resid.std() ** (1/2)}, index=[i])], axis=0)
    
    # Plot the residuals and save as jpg files
    plt.plot(model.resid.rolling(window=60).std( ))
    plt.title(f'{industry_returns.columns[i]}_Adj_5_factor_model')
    plt.savefig(os.path.join('Adjusted_five_factor_model', f'{industry_returns.columns[i]}_Adj_5_factor_model.jpg'))
    plt.close()
adjusted_five_factor_model.to_csv('Adj_five_factor_model.csv')

# Cauculate abosolute intercept divided by absolute of return mean for each indsutry using three factor model, five factor model and adjusted five factor model and save them as one dataframe
# return mean is storaged in Industry_return_summary
Industry_return_summary = pd.read_csv('Industry_returns_summary.csv', index_col=0)
intercept_divided_by_mean = pd.DataFrame()
for i in range(12):
    industry = industry_returns.columns[i]
    intercept_divided_by_mean = pd.concat([intercept_divided_by_mean, pd.DataFrame({'Industry': industry, 'Three_factor_model': abs(three_factor_model.loc[i, 'Intercept']) / abs(Industry_return_summary.loc['mean', industry]), 'Five_factor_model': abs(five_factor_model.loc[i, 'Intercept']) / abs(Industry_return_summary.loc['mean', industry]), 'Adjusted_five_factor_model': abs(adjusted_five_factor_model.loc[i, 'Intercept']) / abs(Industry_return_summary.loc['mean', industry])}, index=[i])], axis=0)

# Use industry names as index
intercept_divided_by_mean.index = industry_returns.columns

# Calculate the mean of intercept_divided_by_mean for each model and add this as a seperate row
intercept_divided_by_mean = pd.concat([intercept_divided_by_mean, pd.DataFrame({'Industry': 'Mean', 'Three_factor_model': intercept_divided_by_mean['Three_factor_model'].mean(), 'Five_factor_model': intercept_divided_by_mean['Five_factor_model'].mean(), 'Adjusted_five_factor_model': intercept_divided_by_mean['Adjusted_five_factor_model'].mean()}, index=['Mean'])], axis=0)
# Add anpther row of mean of intercept_divided_by_mean except BusEq
intercept_divided_by_mean = pd.concat([intercept_divided_by_mean, pd.DataFrame({'Industry': 'Mean_except_BusEq', 'Three_factor_model': intercept_divided_by_mean.loc[intercept_divided_by_mean['Industry'] != 'BusEq', 'Three_factor_model'].mean(), 'Five_factor_model': intercept_divided_by_mean.loc[intercept_divided_by_mean['Industry'] != 'BusEq', 'Five_factor_model'].mean(), 'Adjusted_five_factor_model': intercept_divided_by_mean.loc[intercept_divided_by_mean['Industry'] != 'BusEq', 'Adjusted_five_factor_model'].mean()}, index=['Mean_except_BusEq'])], axis=0)

# Use column industry as index
intercept_divided_by_mean.set_index('Industry', inplace=True)

# And show number in percentage
intercept_divided_by_mean = intercept_divided_by_mean * 100
# Save intercept_divided_by_mean as csv file
intercept_divided_by_mean.to_csv('Intercept_divided_by_mean_percentage.csv')

# Put absolute intercept of three factor model, five factor model and adjusted five factor model and their mean together into one dataframe
intercept = pd.DataFrame()
for i in range(12):
    intercept = pd.concat([intercept, pd.DataFrame({'Industry': industry_returns.columns[i], 'Three_factor_model': abs(three_factor_model.loc[i, 'Intercept']), 'Five_factor_model': abs(five_factor_model.loc[i, 'Intercept']), 'Adjusted_five_factor_model': abs(adjusted_five_factor_model.loc[i, 'Intercept'])}, index=[i])], axis=0)
intercept = pd.concat([intercept, pd.DataFrame({'Industry': 'Mean', 'Three_factor_model': intercept['Three_factor_model'].mean(), 'Five_factor_model': intercept['Five_factor_model'].mean(), 'Adjusted_five_factor_model': intercept['Adjusted_five_factor_model'].mean()}, index=['Mean'])], axis=0)
# Add anothe row of mean of absolute intercept except BusEq
intercept = pd.concat([intercept, pd.DataFrame({'Industry': 'Mean_except_BusEq', 'Three_factor_model': intercept.loc[intercept['Industry'] != 'BusEq', 'Three_factor_model'].mean(), 'Five_factor_model': intercept.loc[intercept['Industry'] != 'BusEq', 'Five_factor_model'].mean(), 'Adjusted_five_factor_model': intercept.loc[intercept['Industry'] != 'BusEq', 'Adjusted_five_factor_model'].mean()}, index=['Mean_except_BusEq'])], axis=0)
intercept.set_index('Industry', inplace=True)
intercept.to_csv('Intercept.csv')



# To calculate factor exposure change around time, I am going to use a rolling window of 60 months to do the linear regression and save all the exposures as a csv file, and I am going to use numpy and OLSanalytical solution to accelerate calculating speed

# Numpy OLS analytical solution with constant
def OLS_analytical_solution(y, x):
    # Add constant to x
    x = sm.add_constant(x)
    # Calculate beta
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return beta

# Create a new folder to store exposures if notb exist
if not os.path.exists('Rolling_exposures'):
    os.mkdir('Rolling_exposures')

# Use this solution to calculate exposures of three factor model, five factor model
for i in range(12):
    industry = industry_returns.columns[i]
    # Five factor model
    exposures = pd.DataFrame()
    for j in range(len(industry_returns) - 61):
        y = industry_returns[industry].iloc[j:j+61]
        x = factor_premiums[['Mkt_minus_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']].iloc[j:j+61]
        # Use the time of the next month of the rolling window as the index of exposures
        exposures = pd.concat([exposures, pd.DataFrame(np.array(OLS_analytical_solution(y, x)).reshape(1, -1), columns= x.columns.to_list() + ["Intercept"], index=[factor_premiums.index[j+61]])], axis=0)

        # exposures = pd.concat([exposures, pd.DataFrame(np.array(OLS_analytical_solution(y, x)).reshape(1, -1), columns=x.columns, index=[x.index[j+60]])], axis=0)
        # use os.path.join to put int folder Rolling_Exposures
    exposures.to_csv(f'Rolling_exposures/Five_factor_model_{industry}.csv')

# Plot the exposures of five factor model for each industry and save as jpg files
for i in range(12):
    industry = industry_returns.columns[i]
    exposures = pd.read_csv(f'Rolling_exposures/Five_factor_model_{industry}.csv', index_col=0)
    # Only show year on x axis but do not change dataframe itself
    exposures.plot(figsize=(20, 10))
    plt.title(f'Five factor model {industry}')
    plt.savefig(f'Rolling_exposures/Five_factor_model_{industry}.jpg')
    plt.close()

# Summury the std of exposures for each industry and save as csv file
exposures_std = pd.DataFrame()
for i in range(12):
    industry = industry_returns.columns[i]
    exposures = pd.read_csv(f'Rolling_exposures/Five_factor_model_{industry}.csv', index_col=0)
    exposures_std = pd.concat([exposures_std, pd.DataFrame({'Industry': industry, 'Mkt_minus_RF': exposures['Mkt_minus_RF'].std(), 'SMB': exposures['SMB'].std(), 'HML': exposures['HML'].std(), 'RMW': exposures['RMW'].std(), 'CMA': exposures['CMA'].std(), 'MOM': exposures['MOM'].std()}, index=[i])], axis=0)
exposures_std = pd.concat([exposures_std, pd.DataFrame({'Industry': 'Mean', 'Mkt_minus_RF': exposures_std['Mkt_minus_RF'].mean(), 'SMB': exposures_std['SMB'].mean(), 'HML': exposures_std['HML'].mean(), 'RMW': exposures_std['RMW'].mean(), 'CMA': exposures_std['CMA'].mean(), 'MOM': exposures_std['MOM'].mean()}, index=['Mean'])], axis=0)
exposures_std.set_index('Industry', inplace=True)
exposures_std.to_csv('Five_factor_model_rolling_exposures_std.csv')


###################################################################################################################################################################
# Question 1.b Deal with colinearity to stabilize the regression and reduce the total error of the estimates


# Use this solution to calculate exposures of three factor model, five factor model
for i in range(12):
    industry = industry_returns.columns[i]
    print(industry)
    # Five factor model
    exposures = pd.DataFrame()
    for j in range(len(industry_returns) - 61):
        y = industry_returns[industry].iloc[j:j+61]
        x = factor_premiums[['Mkt_minus_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']].iloc[j:j+61]

        # # Remove colinearity by regressMSB on RMB and use the residuals as the new SMB 
        # x['SMB'] = sm.OLS(x['SMB'], sm.add_constant(x[['RMW']])).fit().resid
        # x['HML'] = sm.OLS(x['HML'], sm.add_constant(x[['RMW', 'CMA']])).fit().resid

        # Use the time of the next month of the rolling window as the index of exposures use statsmodels to do the regression
        exposures = pd.concat([exposures, pd.DataFrame(np.array(sm.OLS(y, sm.add_constant(x)).fit_regularized(alpha = 5, L1_wt = 0.5).params).reshape(1, -1), columns= x.columns.to_list() + ["Intercept"], index=[factor_premiums.index[j+61]])], axis=0)
                               
                            #    columns=x.columns.to_list() + ["Intercept"], index=[factor_premiums.index[j+61]])], axis=0)

        # exposures = pd.concat([exposures, pd.DataFrame(np.array(OLS_analytical_solution(y, x)).reshape(1, -1), columns=x.columns, index=[x.index[j+60]])], axis=0)
        # use os.path.join to put int folder Rolling_Exposures
    exposures.to_csv(f'Rolling_exposures/Reg_five_factor_model_{industry}.csv')

# Plot the exposures of five factor model for each industry and save as jpg files
for i in range(12):
    industry = industry_returns.columns[i]
    exposures = pd.read_csv(f'Rolling_exposures/Reg_five_factor_model_{industry}.csv', index_col=0)
    # Only show year on x axis but do not change dataframe itself
    exposures.plot(figsize=(20, 10))
    plt.title(f'Reg_five_factor_model_{industry}')
    plt.savefig(f'Rolling_exposures/Reg_five_factor_model_{industry}.jpg')
    plt.close()

# Summury the std of exposures for each industry and save as csv file
exposures_std = pd.DataFrame()
for i in range(12):
    industry = industry_returns.columns[i]
    exposures = pd.read_csv(f'Rolling_exposures/Reg_five_factor_model_{industry}.csv', index_col=0)
    exposures_std = pd.concat([exposures_std, pd.DataFrame({'Industry': industry, 'Mkt_minus_RF': exposures['Mkt_minus_RF'].std(), 'SMB': exposures['SMB'].std(), 'HML': exposures['HML'].std(), 'RMW': exposures['RMW'].std(), 'CMA': exposures['CMA'].std(), 'MOM': exposures['MOM'].std()}, index=[i])], axis=0)
exposures_std = pd.concat([exposures_std, pd.DataFrame({'Industry': 'Mean', 'Mkt_minus_RF': exposures_std['Mkt_minus_RF'].mean(), 'SMB': exposures_std['SMB'].mean(), 'HML': exposures_std['HML'].mean(), 'RMW': exposures_std['RMW'].mean(), 'CMA': exposures_std['CMA'].mean(), 'MOM': exposures_std['MOM'].mean()}, index=['Mean'])], axis=0)
exposures_std.set_index('Industry', inplace=True)
exposures_std.to_csv('Reg_five_factor_model_rolling_exposures_std.csv')







