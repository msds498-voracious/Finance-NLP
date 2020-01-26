# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:57:53 2020

@author: TiwarisUSA
"""

#TextBlob perform simple natural language processing tasks.
#from textblob import TextBlob



import datetime as dt
import math


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tweepy
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import constants as ct
from twitterscraper.query import query_tweets
# =============================================================================
# from tweet import Tweet
# =============================================================================
import oauth_info as auth # our local file with the OAuth infos


style.use('ggplot')
ct.consumer_key = auth.CONSUMER_KEY
ct.consumer_secret = auth.CONSUMER_SECRET
ct.access_token = auth.ACCESS_TOKEN
ct.access_token_secret = auth.ACCESS_TOKEN_SECRET
ct.num_of_tweets = 100
tweet_df_complete = pd.DataFrame()

def retrieving_tweets_polarity(symbol):
    auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
    auth.set_access_token(ct.access_token, ct.access_token_secret)
    user = tweepy.API(auth)

    tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en').items(ct.num_of_tweets)

    tweet_list = []
    global_polarity = 0
    for tweet in tweets:
        tw = tweet.full_text
        print (tw)
        blob = TextBlob(tw)
        polarity = 0
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            print (polarity)
            global_polarity += sentence.sentiment.polarity
            tweet_list.append([tweet.created_at, tw, polarity, symbol])
            
    tweet_df = pd.DataFrame.from_records(tweet_list)    
    global_polarity = global_polarity / len(tweet_list)    
    return tweet_df, global_polarity



if __name__ == "__main__":    
    actual_date = dt.date.today()
    past_date = actual_date - dt.timedelta(days=365 * 3)
    actual_date = actual_date.strftime("%Y-%m-%d")
    past_date = past_date.strftime("%Y-%m-%d")
    for symbol in ['apple', 'facebook', 'netflix', 'google']:
        tweet_df, global_polarity = retrieving_tweets_polarity(symbol)
        tweet_df_complete= pd.concat([tweet_df_complete, tweet_df], axis=0)
        if symbol == 'google':
            tweet_df_complete.to_csv('tweeter_raw_data.csv')
