# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:58:30 2020

@author: TiwarisUSA
"""

import pandas as pd
from dateutil.relativedelta import relativedelta                                                                                                                  
import datetime as dt
import tweepy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import constants as ct
import pickle
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import oauth_info as auth
import quandl
from yahoo_historical import Fetcher
from fbprophet import Prophet
import statsmodels.formula.api as smf
import os.path
# =============================================================================
# Below code trains the Model for the Stock Relevant Classifier
# =============================================================================

df = pd.read_csv('training_dataset.csv')
df.head()

df = df[pd.notnull(df['tweet_text'])]
df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.tweet_text).toarray()
labels = df.category_id
features.shape

N = 2
for tweet_text, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
   

X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['label'], random_state = 0)
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
pickle.dump(count_vect.vocabulary_, open('vector_counts', 'wb'))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# =============================================================================
# Set Date Parameters
# =============================================================================

half = relativedelta(months=6) 
today = pd.Timestamp.today()
six_mo_ago = today - half

five_yrs = relativedelta(months=60) 
today = pd.Timestamp.today()#
five_yrs_ago = today - five_yrs
five_yrs_ago=five_yrs_ago.strftime('%Y-%m-%d')
yr=int(five_yrs_ago[0:4])
mo=int(five_yrs_ago[5:7])
dy=int(five_yrs_ago[8:10])


stock_names = ['aapl', 'amzn', 'fb', 'googl', 'nflx']

# =============================================================================
# Function to read in tweets and sort according to ticker and date parameters
# =============================================================================

def getrawtweets(ticker):
    
    all_tweets = pd.read_csv('all_tweets3.csv')
    top_handles = pd.read_csv('top_names_df.csv')
    top_handle_df = top_handles[top_handles['Stock_Names']==ticker.lower()]
    top_handle_df.rename(columns={'Screen_Name': 'screen_name'}, inplace=True)
    top_handle_list = top_handle_df['screen_name'].tolist()
    all_tweets = all_tweets[all_tweets['screen_name'].isin(top_handle_list)]
    all_tweets['date'] = pd.to_datetime(all_tweets['date'])
    all_tweets = pd.merge(all_tweets,top_handle_df, on='screen_name')
    #all_tweets['date'] = pd.Timestamp(all_tweets['date'])
    
    mask = (all_tweets.date >= six_mo_ago) & (all_tweets.date <= today)
    all_tweets = all_tweets.loc[mask]
    
    all_tweets.drop_duplicates(subset=['date', 'screen_name', 'tweet_text'],
                               keep = False, inplace = True) 
    all_tweets = all_tweets.reset_index(drop=True)
    tweets_comb_stock = pd.DataFrame(columns=('Date', 'Screen Name', 'Sentiment',
                                              'No. of Likes', 'No. of Replies', 'No of Retweets', 'Tweet Text', 'fb',
                                              'aapl', 'amzn', 'nflx', 'googl',
                                              'faang', 'non-faang','Num_Tweets','Tweet_Days', 'Accuracy_30d'))
    tweets_comb_stock[['Date','Screen Name','Sentiment', 'No. of Likes', 'No. of Replies', 'No of Retweets', 'No. of Tweets','No. of Tweet Days','30-Day Return Accuracy','Tweet Text']]=all_tweets[['date','screen_name','polarity', 'likes', 'replies', 'retweets','Num_Tweets','Tweet_Days', 'Accuracy_30d','tweet_text']]
    tweets_comb_stock['fb']=all_tweets['fb']+all_tweets['facebook']
    tweets_comb_stock['aapl']=all_tweets['aapl']+all_tweets['apple']
    tweets_comb_stock['amzn']=all_tweets['amzn']+all_tweets['amazon']
    tweets_comb_stock['nflx']=all_tweets['nflx']+all_tweets['netflix']
    tweets_comb_stock['googl']=all_tweets['googl']+all_tweets['google']
    tweets_comb_stock['faang']=all_tweets['fang']+all_tweets['faang']
    #Create encoding for tweets that don't have a stock reference
    '''for row in range(len(tweets_comb_stock)):
        if tweets_comb_stock.loc[row, ['fb','aapl','amzn','nflx','googl',
                                       'faang']].sum() == 0:
            tweets_comb_stock.loc[row, 'non-faang']=1
        else: 
            tweets_comb_stock.loc[row, 'non-faang']=0
        #print(tweets_comb_stock.loc[row])'''
    
    #New dataframe that holds only FAANG tweets
    #faang_tweets = tweets_comb_stock[tweets_comb_stock['non-faang'] !=1]
    faang_tweets = tweets_comb_stock.sort_values(by=['Date'])
    
    faang_tweets = faang_tweets[faang_tweets[ticker.lower()]==1]
    
    return faang_tweets[['Date','Screen Name','Sentiment','No. of Likes', 'No. of Replies', 'No of Retweets', 'No. of Tweets','No. of Tweet Days','30-Day Return Accuracy','Tweet Text']]


# =============================================================================
# Below code is the actual model. 
# Uncomment the previous lines after the model is trained and run once
# =============================================================================

# Now retreive the tweets for the last 2 days

ct.num_of_tweets = 1000
ct.consumer_key    = '3jmA1BqasLHfItBXj3KnAIGFB'
ct.consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'
ct.access_token  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
ct.access_token_secret = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'

tweet_df_complete = pd.DataFrame()


# =============================================================================
# #Retrieve the tweet relevance model
# filename = 'tweet_classifier.sav'
# tweet_relevance_model = pickle.load(open(filename, 'rb'))
# count_vect = CountVectorizer()
# 
# # Load the vocabulary
# vocabulary_to_load = pickle.load(open('vector_counts', 'r'))
# count_vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,ngram_size), min_df=1, vocabulary=vocabulary_to_load)
# =============================================================================
tweets = pd.read_csv('all_tweets3.csv')

def retrieving_tweets_polarity(ticker_symbol, ticker_symbol_list, name, actual_date, past_date):
    broker_screen_name = name  
    global_polarity = 0
    #tweets = date_filtered_tweet_list_df
    
    tweets_df = tweets
    tweets_df.date = pd.to_datetime(tweets_df['date'], format="%m/%d/%Y %H:%M")
    tweets_df = tweets[(tweets.date < actual_date) & (tweets.date > past_date) & (tweets.screen_name == broker_screen_name)]    
    tweet_list = []
    for index, row in tweets_df.iterrows():   
        #if tweet.created_at < actual_date and tweet.created_at > past_date:
        tw = row[2]          
        blob = TextBlob(tw)
        polarity = 0 
        if any(word in tw.lower() for word in ticker_symbol_list): 
            tweet_relevance = clf.predict(count_vect.transform([tw]))
            if tweet_relevance in ('Yes', 'Maybe'):
                polarity += blob.sentiment.polarity
                print(blob.sentiment.polarity)
                global_polarity += blob.sentiment.polarity
                print(global_polarity)
                tweet_row = [tw, polarity]                    
# =============================================================================
#                 for word in mylist: 
#                     symbol_flag = blob.word_counts[word]
#                     if symbol_flag >= 1: 
#                         tweet_row.append(symbol_flag)
#                     else:                     
#                         tweet_row.append(0)
# =============================================================================
                tweet_list.append(tweet_row)

    tweet_df = pd.DataFrame.from_records(tweet_list)    
    if not tweet_df.empty: 
        global_polarity = global_polarity / len(tweet_list)    
    else: 
        global_polarity = 0
    return global_polarity

# =============================================================================
# Below code is the Financial model
# =============================================================================

quandl.ApiConfig.api_key = auth.QUANDL_KEY

def financial_model(ticker, date):
    date = str(date)
    ticker_df = Fetcher(ticker, [2018,1,1])
    df = ticker_df.get_historical()
    df = df[df.Date < date]
    
    # Keep only the adj. Close 
    adj_close = df[['Date', 'Adj Close']]
    
    # Rename the columns
    adj_close.columns = ['ds', 'y']
    
    # Initialize Prophet instance
    m = Prophet(daily_seasonality=True)
    m.fit(adj_close)
    
    # Since we are forecasting 1, 10 and 30 day returns, lets get the next 30 days. 
    future = m.make_future_dataframe(periods=30)
        
    # Forecast the Prices
    forecast = m.predict(future)
      
    # Lets subset the data to last 30 days
    forecast_30days = forecast[['yhat']].tail(30)
    forecast_30days.reset_index(inplace = True)
    forecast_30days['rows'] = forecast_30days.index
    
    # Fit regression model (using the natural log of one of the regressors)
    results_30 = smf.ols('yhat ~ rows', data=forecast_30days).fit()

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
    return results_30.params[1], normalized_slope

from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = date(2019, 7, 1)
end_date = date(2019, 12, 31)
myList = []
ticker = 'GOOGL'

for single_date in daterange(start_date, end_date):
    print(single_date)
    slope, norm_slope = financial_model(ticker, single_date.strftime("%Y-%m-%d"))
    # Getting the top handles for the each of the desired ticker
    top_handles = getrawtweets(ticker)
    
    # Filter on the tweets
    top_handles.sort_values(by=['30-Day Return Accuracy'], ascending = False,  inplace = True)
    top_handles = top_handles[top_handles['30-Day Return Accuracy'] >= 0.5]
    top_handles = top_handles[top_handles['No. of Tweets'] >= 4]
    
    # Get the unique handles
    top_handles = top_handles['Screen Name'].unique()
    # Get the tweet polarity 
    if ticker == 'GOOGL': 
        mylist = ['google', 'googl', 'fang', 'faang']
    elif ticker == 'APPL': 
        mylist = ['appl', 'apple', 'fang', 'faang']
    elif ticker == 'FB': 
        mylist = ['fb', 'facebook', 'fang', 'faang']
    elif ticker == 'AMZN': 
        mylist = ['amzn', 'amazon', 'fang', 'faang']
    else : 
        mylist = ['nflx', 'netflix', 'fang', 'faang']
    
    actual_date = single_date
    past_date = actual_date - dt.timedelta(days=2)
    actual_date = datetime.datetime.combine(actual_date, datetime.time(0, 0))
    past_date = datetime.datetime.combine(past_date, datetime.time(0, 0))
    stock_broker_list = top_handles
    total_polarity = 0
    for broker_name in stock_broker_list: 
        global_polarity = retrieving_tweets_polarity(ticker, mylist, broker_name, actual_date,
                                                               past_date)
        total_polarity += global_polarity/ len(stock_broker_list)
    
    #Now construct the data rows
    data_row = [ticker, single_date, slope, norm_slope, total_polarity]
    myList.append(data_row)

# Validation
validation_df = pd.DataFrame(myList)
validation_df.columns = ['ticker', 'date', 'slope', 'norm_slope', 'total_polarity']

conditions = [
    (validation_df['total_polarity'] >= 0.5),
    (validation_df['total_polarity'] >= -0.5) & (validation_df['total_polarity'] < 0.5),
    (validation_df['total_polarity'] <= -0.5)]
choices = [1, 2, 3]
validation_df['norm_polarity'] = np.select(conditions, choices)
validation_df['slope_polarity'] = validation_df['norm_slope'] * validation_df['norm_polarity']


conditions = [
    (validation_df['slope_polarity'] >= 10),
    (validation_df['slope_polarity'] >= 8) & (validation_df['slope_polarity'] < 10),
    (validation_df['slope_polarity'] >= 5) & (validation_df['slope_polarity'] < 8),
    (validation_df['slope_polarity'] >= 4) & (validation_df['slope_polarity'] < 5),
    (validation_df['slope_polarity'] < 4)]
choices = ['Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell']
validation_df['Recommendation'] = np.select(conditions, choices)

#Merge prices with validation_df
prices_df = Fetcher(ticker, [2018,1,1])
prices_df = prices_df.get_historical()
prices_df = prices_df[['Date', 'Adj Close']]

#Merge with the validation data
validation_df.merge(prices_df, left_on = 'date', right_on= 'Date', how='left')
