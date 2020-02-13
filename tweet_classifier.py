# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:22:47 2020

@author: TiwarisUSA
"""

import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
  print("# '{}':".format(tweet_text))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  

X_train, X_test, y_train, y_test = train_test_split(df['tweet_text'], df['label'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

#lets classify all our data
mydata = pd.read_csv('all_tweets3.csv')
mydata.head()
mydata_texts = mydata[['tweet_text']]
final_result = []
for text in mydata_texts.tweet_text: 
    print(text)
    result = clf.predict(count_vect.transform([text]))
    print(result)
    final_result.append(result)
    
final_result_df= pd.DataFrame(final_result)


