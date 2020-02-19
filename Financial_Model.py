# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:58:30 2020

@author: TiwarisUSA
"""

import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import oauth_info as auth
import quandl
from yahoo_historical import Fetcher
from fbprophet import Prophet
import statsmodels.api as sm
import statsmodels.formula.api as smf

quandl.ApiConfig.api_key = auth.QUANDL_KEY

ticker_googl = Fetcher("GOOGL", [2010,1,1])
googl_df=ticker_googl.get_historical()
print(googl_df.head())

# The adjusted close accounts for stock splits, so that is what we should graph
plt.plot(googl_df.Date, googl_df['Adj Close'])
plt.title('Google Stock Price')
plt.ylabel('Price ($)');
plt.show()

# Keep only the adj. Close 
googl_adj_close = googl_df[['Date', 'Adj Close']]

# Rename the columns
googl_adj_close.columns = ['ds', 'y']

# Initialize Prophet instance
m = Prophet()
m.fit(googl_adj_close)

# Since we are forecasting 1, 10 and 30 day returns, lets get the next 30 days. 
future = m.make_future_dataframe(periods=30)
future.tail()

# Forecast the Prices
forecast = m.predict(future)

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

# Lets subset the data to last 30 days
forecast_30days = forecast[['yhat']].tail(30)
forecast_30days.reset_index(inplace = True)
forecast_30days['rows'] = forecast_30days.index

# Fit regression model (using the natural log of one of the regressors)
results = smf.ols('yhat ~ rows', data=forecast_30days).fit()

print(results.summary())
print(results.params[1])

if results.params[1] > 0.5: 
    print ('Agressive rise')
elif 0 < results.params[1] <= 0.5: 
    print('Average Rise')
elif -0.5 < results.params[1] <= 0:
    print('Average Decline')
else: 
    print('Aggressive decline')
