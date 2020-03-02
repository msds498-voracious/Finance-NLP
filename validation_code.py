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

ticker_googl = Fetcher("NFLX", [2018,1,1])
googl_df=ticker_googl.get_historical()
print(googl_df.head())

# The adjusted close accounts for stock splits, so that is what we should graph
plt.plot(googl_df.Date, googl_df['Adj Close'])
plt.title('Google Stock Price')
plt.ylabel('Price ($)');
plt.xticks(rotation=45)
plt.show()

# Keep only the adj. Close 
googl_adj_close = googl_df[['Date', 'Adj Close']]

# Rename the columns
googl_adj_close.columns = ['ds', 'y']

# Initialize Prophet instance
m = Prophet(daily_seasonality=True)
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

# Lets subset the data to last 10 days
# =============================================================================
# forecast_10days = forecast[['yhat']].tail(10)
# forecast_10days.reset_index(inplace = True)
# forecast_10days['rows'] = forecast_10days.index
# 
# # Lets subset the data to last 30 days
# forecast_1day = forecast[['yhat']].tail(1)
# forecast_1day.reset_index(inplace = True)
# forecast_1day['rows'] = forecast_1day.yhat.iloc[0] - forecast_1day.index
# =============================================================================

# Fit regression model (using the natural log of one of the regressors)
results_30 = smf.ols('yhat ~ rows', data=forecast_30days).fit()
# =============================================================================
# results_10 = smf.ols('yhat ~ rows', data=forecast_10days).fit()
# results_1 = (forecast_1day.yhat.iloc[0] - googl_adj_close.y.iloc[-1]) / googl_adj_close.y.iloc[-1]
# =============================================================================

#print(results.summary())
print(results_30.params[1])
# =============================================================================
# print(results_10.params[1])
# print(results_1)
# =============================================================================

if results_30.params[1] > 2: 
    normalized_slope = 5
elif 0.5 < results_30.params[1] <= 2: 
    normalized_slope = 4
elif -0.5 < results_30.params[1] <= 0.5:
    normalized_slope = 3
elif -2 < results_30.params[1] <= -0.5:
    normalized_slope = 2
else: 
    normalized_slope = 1


