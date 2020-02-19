# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:00:53 2020

@author: DELL
"""

import pandas as pd
#from datetime import timedelta 
from dateutil.relativedelta import relativedelta                                                                                                                  
from datetime import date 

def getrawtweets(ticker):
    
    all_tweets = pd.read_csv('G:/My Drive/MSPA/MSDS 498 - Capstone/Data/Finance-NLP-master/all_tweets3.csv')
    all_tweets['date'] = pd.to_datetime(all_tweets['date'])
    
    half = relativedelta(months=6) 
    today = date.today()
    six_mo_ago = today - half
    
    mask = (all_tweets.date >= six_mo_ago) & (all_tweets.date <= today)
    all_tweets = all_tweets.loc[mask]
    
    all_tweets.drop_duplicates(subset=['date', 'screen_name', 'tweet_text'],
                               keep = False, inplace = True) 
    all_tweets = all_tweets.reset_index(drop=True)
    tweets_comb_stock = pd.DataFrame(columns=('Date', 'Screen Name', 'Sentiment',
                                              'No. of Likes', 'No. of Replies', 'No of Retweets', 'Tweet Text', 'fb',
                                              'aapl', 'amzn', 'nflx', 'googl',
                                              'faang', 'non-faang'))
    tweets_comb_stock[['Date','Screen Name','Sentiment', 'No. of Likes', 'No. of Replies', 'No of Retweets','Tweet Text']]=all_tweets[['date','screen_name','polarity', 'likes', 'replies', 'retweets','tweet_text']]
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
    
    return faang_tweets[['Date','Screen Name','Sentiment','No. of Likes', 'No. of Replies', 'No of Retweets','Tweet Text']]
