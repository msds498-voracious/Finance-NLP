# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:00:53 2020
@author: TiwarisUSA
"""

import pandas as pd
from dateutil.relativedelta import relativedelta                                                                                                                  
import datetime as dt
import math
import tweepy
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import constants as ct
from twitterscraper.query import query_tweets
import pickle
import datetime
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('training_dataset.csv')
df.head()

df = df[pd.notnull(df['tweet_text'])]
df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)
df.head()

fig = plt.figure(figsize=(8,6))
df.groupby('label').tweet_text.count().plot.bar(ylim=0)
plt.show()

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
    
    all_tweets = pd.read_csv('https://raw.githubusercontent.com/msds498-voracious/Dashboard/master/all_tweets3.csv')
    #all_tweets = pd.read_csv('https://raw.githubusercontent.com/msds498-voracious/Finance-NLP/master/faang_tweets.csv')
    top_handles = pd.read_csv('https://raw.githubusercontent.com/msds498-voracious/Dashboard/master/top_names_df.csv')
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


df = getrawtweets('FB')

# Filter on the tweets
df.sort_values(by=['30-Day Return Accuracy'], ascending = False,  inplace = True)
df = df[df['30-Day Return Accuracy'] >= 0.5]
df = df[df['No. of Tweets'] >= 4]

# Get the unique handles
top_handles = df['Screen Name'].unique()

# Now retreive the tweets for the last 10 days

ct.num_of_tweets = 1000
ct.consumer_key    = '3jmA1BqasLHfItBXj3KnAIGFB'
ct.consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'
ct.access_token  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
ct.access_token_secret = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'

tweet_df_complete = pd.DataFrame()
mylist = ['facebook', 'fb', 'fang', 'faang']

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

def retrieving_tweets_polarity(name):
    broker_screen_name = name
    auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
    auth.set_access_token(ct.access_token, ct.access_token_secret)
    user = tweepy.API(auth, wait_on_rate_limit=True)
    tweets = tweepy.Cursor(user.user_timeline, screen_name = broker_screen_name, tweet_mode='extended', lang='en').items(ct.num_of_tweets)
    tweet_list = []
    global_polarity = 0
    for tweet in tweets:   
        if tweet.created_at < actual_date and tweet.created_at > past_date:
            tw = tweet.full_text           
            blob = TextBlob(tw)
            polarity = 0 
            if any(word in tw.lower() for word in mylist): 
                tweet_relevance = clf.predict(count_vect.transform([tw]))
                print(tweet_relevance)
                if tweet_relevance in ('Yes', 'Maybe'):
                    print('i am in')
                    polarity += blob.sentiment.polarity
                    global_polarity += blob.sentiment.polarity
                    tweet_row = [tweet.created_at,broker_screen_name, tw, polarity]                    
                    for word in mylist: 
                        symbol_flag = blob.word_counts[word]
                        if symbol_flag >= 1: 
                            tweet_row.append(symbol_flag)
                        else:                     
                            tweet_row.append(0)
                    tweet_list.append(tweet_row)
        else: 
           pass
    tweet_df = pd.DataFrame.from_records(tweet_list)    
    if not tweet_df.empty: 
        global_polarity = global_polarity / len(tweet_list)    
    else: 
        global_polarity = 0
    return tweet_df, global_polarity

   
actual_date = dt.date.today()
past_date = actual_date - dt.timedelta(days=30)
actual_date = datetime.datetime.combine(actual_date, datetime.time(0, 0))
past_date = datetime.datetime.combine(past_date, datetime.time(0, 0))
stock_broker_list = top_handles
total_polarity = 0
for broker_name in stock_broker_list: 
    tweet_df, global_polarity = retrieving_tweets_polarity(broker_name)
    total_polarity += global_polarity/ len(stock_broker_list)
    tweet_df_complete= pd.concat([tweet_df_complete, tweet_df], axis=0)    
tweet_df_complete.to_csv('relevant_tweet_data.csv')

                   