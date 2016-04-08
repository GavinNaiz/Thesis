# -*- coding: utf-8 -*-

import sys, ast, re, json
from random import shuffle
import numpy as np
import features
import logit
from texttable import Texttable



doc = json.loads(open('convs_imb.json').read())
keys_bal = np.load('keys_bal.npy')
keys_imb = np.load('keys_imb.npy')

postags_bal = open('postags_bal.txt').read().split('\n')
postags_imb = open('postags_imb.txt').read().split('\n')

def clean(text):
    """ remove user names and URLS """
    cleaned_text = ' '
    for word in text.split():
        if word[0] == '@':
            cleaned_text += 'USERNAME' + ' '
        elif word[:4] == 'http':
            cleaned_text += 'URL' + ' '
        else:
            cleaned_text += word + ' '
    return cleaned_text

labels = []
texts = []
author_profiles = []
auth_prof_descs = []
aud_prof_descs = []
audience_profiles = []
hist_comms = []
pairwise_tweets = []
original_texts = []

#for key in keys_imb:
    #labels.append(doc[str(key)]['label'])
    
    #texts.append(clean(doc[str(key)]['author']['text']))
    #author_profiles.append([doc[str(key)]['author']['user']['name'].split()[0], doc[str(key)]['author']['user']['friends_count'], doc[str(key)]['author']['user']['followers_count'], doc[str(key)]['author']['user']['statuses_count'], doc[str(key)]['author']['user']['created_at'], doc[str(key)]['author']['created_at'][4:10], doc[str(key)]['author']['user']['time_zone'], doc[str(key)]['author']['user']['verified']]) 
    #if doc[str(key)]['author']['user']['description'] != None:
     #   auth_prof_descs.append(clean(doc[str(key)]['author']['user']['description']))
    #else:
     #   auth_prof_descs.append(' ')

    #if doc[str(key)]['audience']['user']['description'] != None:
     #   aud_prof_descs.append(clean(doc[str(key)]['audience']['user']['description']))
    #else:
     #   aud_prof_descs.append(' ')
    #audience_profiles.append([doc[str(key)]['audience']['user']['name'].split()[0], doc[str(key)]['audience']['user']['friends_count'], doc[str(key)]['audience']['user']['followers_count'], doc[str(key)]['audience']['user']['statuses_count'], doc[str(key)]['audience']['user']['created_at'], doc[str(key)]['audience']['created_at'][4:10], doc[str(key)]['audience']['user']['time_zone'], doc[str(key)]['audience']['user']['verified']]) 
    #hist_comms.append({'aud': doc[str(key)]['audience']['user'], 'auth': doc[str(key)]['author']['user']})

    #conv = [doc[str(key)]['author']['text']]
    #conv.append(doc[str(key)]['audience']['text'])
    #pairwise_tweets.append(conv)
    #original_texts.append(clean(doc[str(key)]['audience']['text']))


#y_array = labels

#X_unigrams = features.ngram_feats(texts, "uni").todense()
#X_bigrams = features.ngram_feats(texts, "bi").todense()
#X_brown_uni = features.ngram_feats(features.brown_features(texts), "uni").todense()
#X_brown_bi = features.ngram_feats(features.brown_features(texts), "bi").todense()
#X_pron = features.pronunciation_feats(texts)
#X_cap = features.capital_feats(texts)
#X_intense = features.intensifiers(texts, 2240)
X_pos = features.pos_features(postags_imb)

#X_auth_profs = features.profile_info(author_profiles)
#X_auth_descs = features.ngram_feats(auth_prof_descs, "uni").todense()

#X_aud_descs = features.ngram_feats(aud_prof_descs, "uni").todense()
#X_aud_profs = features.profile_info(audience_profiles)
#X_hist_comms = features.communication_feats(hist_comms)

#X_pairwise_brown_bal = features.make_pairwise_brown(pairwise_tweets) # MAKE PAIRWISE DICTIONARY
#X_pairwise_brown = features.pairwise_brown(pairwise_tweets)
#X_orig_unigrams = features.ngram_feats(original_texts, "uni").todense()

#np.save('y_array_imb', y_array)

#np.save('X_unigram_imb.npy', X_unigrams)
#np.save('X_bigram_imb.npy', X_bigrams)
#np.save('X_brown_uni_imb.npy', X_brown_uni)
#np.save('X_brown_bi_imb.npy', X_brown_bi)
#np.save('X_pron_imb.npy', X_pron)
#np.save('X_cap_bal.npy', X_cap)
#np.save('X_intense_imb.npy', X_intense)
np.save('X_pos_imb.npy', X_pos)

#np.save('X_auth_profs_imb', X_auth_profs)
#np.save('X_auth_descs_imb', X_auth_descs)

#np.save('X_aud_descs_imb', X_aud_descs)
#np.save('X_aud_profs_imb', X_aud_profs)
#np.save('X_hist_comms_bal', X_hist_comms)

#np.save('X_pairwise_brown_imb', X_pairwise_brown)
#np.save('X_orig_unigram_imb.npy', X_orig_unigrams)


