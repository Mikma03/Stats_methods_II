import pandas as pd
import numpy as np
import statsmodels
from matplotlib import pyplot as plt

df = pd.read_csv(r'C:\Users\mikol\Desktop\Python_Met_II/monthly_csv.csv', header=0, infer_datetime_format=True, parse_dates=[0],
                 index_col=[0])
print(df.head(10))

df['Time_Period'] = range(1, len(df)+1)
#Print the first 10 rows:

print(df.head(10))

# Create a new pyplot figure to plot into
fig = plt.figure()

# Set the title of the plot
fig.suptitle('Export Price Index of Gold')

# Set the X and Y axis labels
plt.xlabel('Time Period')
plt.ylabel('Price Index')

# plot the time series and store the plot in the actual variable. We'll need that later for the legend.
actual, = plt.plot(df['Time_Period'], df['Price'], 'bo-', label='Gold Price Index')

# Set up the legend. There is only one time series in the legend.
plt.legend(handles=[actual])

# Show everything
plt.show()

# create a time lagged column
df['LAGGED_Export_Price_Index_of_Gold'] = df['Price'].shift(1)

# Do a diff between the Export_Price_Index_of_Gold column and the time lagged version
df['DIFF_Export_Price_Index_of_Gold'] = df['Price'] - df['LAGGED_Export_Price_Index_of_Gold']

# Plot the diff column using Series.plot()
df['DIFF_Export_Price_Index_of_Gold'].plot()

# Display the plot
plt.show()




df['LOG_Export_Price_Index_of_Gold'] = np.log(df['Price'])

# Create a (2 x 1) grid of subplots
ax = plt.subplot(1, 2, 1)

# Set the title of the first sub-plot
ax.set_title('Export_Price_Index_of_Gold versus Time_Period', fontdict={'fontsize': 12})

# Plot the RAW scale plot
plt.scatter(x=df['Time_Period'].values, y=df['Price'].values, color='r', marker='.')

# Setup the second subplot
ax = plt.subplot(1, 2, 2)

ax.set_title('LOG(Price) versus Time_Period', fontdict={'fontsize': 12})

# Plot the LOG scale plot
plt.scatter(x=df['Time_Period'].values, y=df['LOG_Export_Price_Index_of_Gold'].values, color='b', marker='.')

# Display both subplots
plt.show()




import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

expr = 'LOG_Export_Price_Index_of_Gold ~ Time_Period'
split_index = 800

# Get the indexed date at the split position
split_date = df.index[split_index]

# time periods 0 to 119 is the training set
df_train = df.loc[df.index <= split_date].copy()

# time periods 120 to 131 is the testing set
df_test = df.loc[df.index > split_date].copy()

print('Model will train on the first ' + str(len(df_train)) + ' months and make predictions for the final ' + str(
    len(df_test)) + ' months.')

olsr_results = smf.ols(expr, df_train).fit()
print(olsr_results.summary())


olsr_predictions_training_set = olsr_results.predict(df_train['Time_Period'])

# Predict log(y) on the testing data set
olsr_predictions_testing_set = olsr_results.predict(df_test['Time_Period'])

# Create the pyplot figure for plotting
fig = plt.figure()

fig.suptitle('Predicted versus actual values of LOG(price index)')

# Plot the log-scale PREDICTIONS for the training data set
predicted_training_set, = plt.plot(df_train.index, olsr_predictions_training_set, 'go-',
                                   label='Predicted (Training data set)')

# Plot the log-scale ACTUALS fpr the training data set
actual_training_set, = plt.plot(df_train.index, df_train['LOG_Export_Price_Index_of_Gold'], 'ro-',
                                label='Actuals (Training data set)')

# Plot the log-scale PREDICTIONS for the testing data set
predicted_testing_set, = plt.plot(df_test.index, olsr_predictions_testing_set, 'bo-',
                                  label='Predicted (Testing data set)')

# Plot the log-scale ACTUALS for the testing data set
actual_testing_set, = plt.plot(df_test.index, df_test['LOG_Export_Price_Index_of_Gold'], 'mo-',
                               label='Actuals (Testing data set)')

# Set up the legends
plt.legend(handles=[predicted_training_set, actual_training_set, predicted_testing_set, actual_testing_set])

# Display everything
plt.show()


plt.hist(olsr_results.resid, bins=100)
plt.show()

print('Mean of residual errors='+str(olsr_results.resid.mean()))
print('Variance of residual errors='+str(olsr_results.resid.var()))


expr = 'LOG_Export_Price_Index_of_Gold ~ Time_Period'
y_train, X_train = dmatrices(expr, df_train, return_type='dataframe')

from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip

keys = ['Lagrange Multiplier statistic:', 'LM test\'s p-value:', 'F-statistic:', 'F-test\'s p-value:']
test = het_white(olsr_results.resid, X_train)
lzip(keys, test)


olsr_predictions_raw_scale_training_set = np.exp(olsr_predictions_training_set)
actuals_raw_scale_training_set = np.exp(df_train['LOG_Export_Price_Index_of_Gold'])
olsr_predictions_raw_scale_testing_set = np.exp(olsr_predictions_testing_set)
actuals_raw_scale_testing_set = np.exp(df_test['LOG_Export_Price_Index_of_Gold'])

adjusted_olsr_predictions_raw_scale_training_set = olsr_predictions_raw_scale_training_set*1.01
adjusted_olsr_predictions_raw_scale_testing_set = olsr_predictions_raw_scale_testing_set*1.01

# Create the pyplot figure for plotting
fig = plt.figure()
fig.suptitle('Predicted versus actual values of Gold Price Index')

# Plot the raw scale predictions made on the training data set
predicted_training_set, = plt.plot(df_train.index, adjusted_olsr_predictions_raw_scale_training_set, 'go-',
                                   label='Predicted (Training data set)')

# Plot the raw scale predictions made on the testing data set
predicted_testing_set, = plt.plot(df_test.index, adjusted_olsr_predictions_raw_scale_testing_set, 'bo-',
                                  label='Predicted (Testing data set)')

# Plot the raw scale actual values in the testing data set
actual_testing_set, = plt.plot(df_test.index, df_test['Price'], 'mo-',
                               label='Actuals (Testing data set)')

# Set up the legends
plt.legend(handles=[predicted_training_set, actual_training_set, predicted_testing_set, actual_testing_set])

# Display everything
plt.show()





