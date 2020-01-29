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

ct.consumer_key    = '3jmA1BqasLHfItBXj3KnAIGFB'
ct.consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'
ct.access_token  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
ct.access_token_secret = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'

style.use('ggplot')

ct.num_of_tweets = 1000
tweet_df_complete = pd.DataFrame()
mylist = ['apple', 'facebook', 'netflix', 'google', 'fb', 'appl', 'amzn', 'nflx', 'googl', 'amazon' , 'fang', 'faang']

def retrieving_tweets_polarity(name):
    broker_screen_name = '@' + name
    auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
    auth.set_access_token(ct.access_token, ct.access_token_secret)
    user = tweepy.API(auth, wait_on_rate_limit=True)
    tweets = tweepy.Cursor(user.user_timeline, screen_name = broker_screen_name, tweet_mode='extended', lang='en').items(ct.num_of_tweets)
    tweet_list = []
    global_polarity = 0
    for tweet in tweets:        
        tw = tweet.full_text          
        blob = TextBlob(tw)
        polarity = 0 
        if any(word in tw.lower() for word in mylist):            
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

if __name__ == "__main__":    
    actual_date = dt.date.today()
    past_date = actual_date - dt.timedelta(days=365 * 3)
    actual_date = actual_date.strftime("%Y-%m-%d")
    past_date = past_date.strftime("%Y-%m-%d")
    stock_broker_list =pd.read_csv('micheal_2nd_list.csv', header = None)
    stock_broker_list.columns = ['broker_names']    
    for broker_name in stock_broker_list.broker_names: 
        tweet_df, global_polarity = retrieving_tweets_polarity(broker_name)
        tweet_df_complete= pd.concat([tweet_df_complete, tweet_df], axis=0)    
    tweet_df_complete.columns = ['date', 'screen_name', 'tweet_text', 'polarity', ' apple ', ' facebook ', ' netflix ', ' google ', ' fb ', ' appl ', ' amzn ', ' nflx ', ' googl ', ' amazon ' , ' fang ', ' faang ']
    tweet_df_complete.to_csv('all_tweets.csv')
