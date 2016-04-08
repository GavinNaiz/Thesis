# -*- coding: utf-8 -*-

import json
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from sklearn import linear_model, metrics


#keys_bal = np.load('keys_bal.npy')
#keys_imb = np.load('keys_imb.npy')
tweets = json.load(open('convs_imb.json'))


#salient_terms = []

def clean(text):

    cleaned_text = []
    for tweet in text.split('\n'):
        cleaned_tweet = ' '
        for word in tweet.split():
            if word[0] == '@':
                cleaned_tweet += 'USERNAME' + ' '
            elif word[:4] == 'http':
                cleaned_tweet += 'URL' + ' '
            else:
                cleaned_tweet += word + ' '
        cleaned_text.append(cleaned_tweet)
    return cleaned_text

def get_salient_set(keys):
    salient_terms = []
    tfidf = TfidfVectorizer(stop_words='english')
    top_n = 100
    #keys = np.hstack((keys_bal[:split_start], keys_bal[split_end:]))#keys_bal[:717]#np.hstack((keys_bal[:358], keys_bal[538:]))
    for key in keys: #717 179
        history = clean(tweets[str(key)]['audience']['user']['history'])
        try:
            teeeff = tfidf.fit_transform(history)
            indices = np.argsort(tfidf.idf_)[::-1]
            features = tfidf.get_feature_names()
            top_terms = [features[i] for i in indices[:top_n]]
            for t in top_terms:
                salient_terms.append(t)
        except:
            continue
        
    salient_set = list(set(salient_terms))
    return salient_set




def get_salience_matrix(keys, salient_set):
    """ run test set on salient terms """
    salient_feats = []
    tfidf = TfidfVectorizer(stop_words='english')
    top_n = 100
    for key in keys:
        salience_test = []
        top_terms = []
        history = clean(tweets[str(key)]['audience']['user']['history'])[1:]
        #print len(history)
        try:
            teeeff = tfidf.fit_transform(history)
            indices = np.argsort(tfidf.idf_)[::-1]
            features = tfidf.get_feature_names()
            top_terms = [features[i] for i in indices[:top_n]]
        except:
            top_terms = []

        for term in salient_set:
            if term in top_terms:
                salience_test.append(1)
            else:
                salience_test.append(0)
        salient_feats.append(salience_test)
    return np.array(salient_feats)
#print len(salient_feats)


"""y_array = np.load('y_array_imb.npy')
X_array = np.array(salient_feats)

#print y_array.shape, X_array.shape

# divide train/test sets
X_train = np.vstack((X_array[:448,:], X_array[896:,:]))
X_test = X_array[448:896]
y_train = np.hstack((y_array[:448], y_array[896:]))#np.hstack((y_array[:358], y_array[538:]))
y_test = y_array[448:896]#[358:538]

print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

logistic = linear_model.LogisticRegression(class_weight="balanced")

acc = logistic.fit(X_train, y_train).score(X_test, y_test)
pred = logistic.predict(X_test)
f1 = metrics.f1_score(y_test, pred)
pres = metrics.precision_score(y_test, pred)
rec = metrics.recall_score(y_test, pred)

print acc, pres, rec, f1"""

