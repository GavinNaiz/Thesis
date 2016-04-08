# -*- coding: utf-8 -*-

import tweepy
import json
from random import shuffle
import numpy as np

"""consumer_key = 'ZSwIyb7gDtEv4h3j415u1Qng5'
consumer_secret = 'NuiQFevln0nC9JuEGBuhdoP4mketh4SxJ6wtg6TvqPi4eu0hBN'
access_token_key = '20521933-TF8iiSRLUOcA28Sss4969Hbm3bgmLXl5Oj75SqBOc'
access_token_secret = 'fKJAiMteJkeZgDEEQWmmdqTpvxo8yTqX5iFwPVwdYw06S'

#def get_hashed_tweets():

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
api = tweepy.API(auth)

hashed_tweets = {}

tweet_key = 2131

#for tweet in tweepy.Cursor(api.search, q=('"#sarcasm"'), lang='en').items(448):
for tweet in tweepy.Cursor(api.search, q=('" "'), lang='en').items(2240):
    hashed_tweets[tweet_key] = tweet._json
    tweet_key += 1

with open('hash_neg11.json', 'w') as outfile:
    json.dump(hashed_tweets, outfile)"""
#####################################################
#hash_pos = json.load(open('hash_pos1.json'))
#hash_pos = json.load(open('hash_pos2.json'))
#hash_pos = json.load(open('hash_pos3.json'))
#hash_pos = json.load(open('hash_pos4.json'))
#hash_pos = json.load(open('hash_pos5.json'))
#hash_pos = json.load(open('hash_pos6.json')) # len = 517
hash_pos = json.load(open('hash_pos7.json')) # 2240

new_hash_pos = {}
key = 0
for k, v in hash_pos.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hash_pos[key] = v
        key += 1
print len(new_hash_pos)

#####################################################
""" # add all negs together
hashn1 = json.load(open('hash_neg1.json'))
new_hashn = {}
key = 448
for k, v in hashn1.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn2 = json.load(open('hash_neg2.json'))
key = 655
for k, v in hashn2.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn3 = json.load(open('hash_neg3.json'))
key = 891
for k, v in hashn3.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn4 = json.load(open('hash_neg4.json'))
key = 1144
for k, v in hashn4.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn5 = json.load(open('hash_neg5.json'))
key = 1384
for k, v in hashn5.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn6 = json.load(open('hash_neg6.json'))
key = 1629
for k, v in hashn6.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn7 = json.load(open('hash_neg7.json'))
key = 1830
for k, v in hashn7.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn8 = json.load(open('hash_neg8.json'))
key = 2085
for k, v in hashn8.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)"""

"""hashn9 = json.load(open('hash_neg9.json'))
key = 2348
for k, v in hashn9.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)

hashn10 = json.load(open('hash_neg10.json'))
key = 3348
for k, v in hashn10.iteritems():
    if v['in_reply_to_status_id'] != None:
        new_hashn[key] = v
        key += 1
print len(new_hashn)"""

"""new_new_hashn = {}

for k, v in new_hashn.iteritems():
    #if k < 2240:
    new_new_hashn[k] = v

print len(new_new_hashn)

with open('hash_neg_new.json', 'w') as outfile:
    json.dump(new_new_hashn, outfile)"""

#####################################################
"""with open('hash_pos1_new.json', 'w') as outfile:
    json.dump(new_hashp, outfile)"""

#hash_pos = json.load(open('hash_pos1_new.json'))

new_new_hash_pos = {}

print len(new_hash_pos)

#for key in range(448, 541):
 #   del hash_pos[str(key)]


""" get rid of #sarcasm """
for k, v in new_hash_pos.iteritems():
    v['text'] = v['text'].replace('#sarcasm', " ")
    v['text'] = v['text'].replace('#Sarcasm', " ")
    v['text'] = v['text'].replace('#SARCASM', " ")
    new_new_hash_pos[k] = v

print len(new_new_hash_pos)

with open('hash_pos_new.json', 'w') as outfile:
    json.dump(new_new_hash_pos, outfile)

###################################################3

"""### combine pos + neg

hash_pos = json.load(open('hash_pos1_new.json'))
hash_neg = json.load(open('hash_neg_new.json'))

convs_hash_imb = {}

for key, val in hash_pos.iteritems():
    val['label'] = 1
    convs_hash_imb[key] = {'author': val}

print len(convs_hash_imb)

for key, val in hash_neg.iteritems():
    val['label'] = 0
    convs_hash_imb[key] = {'author': val}

print len(convs_hash_imb)

with open('convs_hash_imb.json', 'w') as outfile:
    json.dump(convs_hash_imb, outfile)"""

##################################################
# create keys list
"""keys_hash_imb = list(xrange(2240))
shuffle(keys_hash_imb)

np.save('keys_hash_imb.npy', keys_hash_imb)"""

