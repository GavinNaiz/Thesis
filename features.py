# -*- coding: utf-8 -*-

from __future__ import division
import sys, re, operator
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json
from itertools import groupby
from collections import Counter
import itertools



def ngram_feats(document, gram):
    """ vectorize and create BOW model"""              
    if gram == 'uni':
        vectorizer = CountVectorizer() 
        vected = vectorizer.fit_transform(document)
    elif gram == 'bi':
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)  
        vected = bigram_vectorizer.fit_transform(document)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(vected)

    return tfidf

def pronunciation_feats(tweets):
    """ 2 features: words with only alphabetic characters but no vowels (e.g., btw) and the number of words with more than three syllables"""
    pron_feats = []
    for tweet in tweets:
        pf = [] # pron feats for each tweet
        cnst = 0 # no of consonant only words
        syl4 = 0 # no of high syllable words
        for word in tweet.split():
            if re.search('(?i)^[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]+$', word):
                cnst += 1
            """ find words of 4+ syllables """
            sc = 0
            count = 0
            vowels = 'aeiouy'
            word = word.lower().strip(".:;?!")
            if len(word) > 0:
                if word[0] in vowels:
                    count +=1

                for index in range(1,len(word)):
                    if word[index] in vowels and word[index-1] not in vowels:
                        count +=1
                if word.endswith('e'):
                    count -= 1
                if word.endswith('le'):
                    count+=1
                if count == 0:
                    count +=1
            if count > 3: # if more than 3 syllables
                syl4 += 1
        pf.append(cnst/5.0) # normalised by max orrurence
        pf.append(sc/5.0)
        pron_feats.append(pf)
    return np.array(pron_feats)

def capital_feats(tweets):
    cap_feats = []
    max_in_cap = 0
    max_all_cap = 0
    for tweet in tweets:
        cf = []
        in_cap = 0 # initial capital
        all_cap = 0 # all caps (two or more letters)
        for word in tweet.split():
            if word != 'USERNAME' and word != 'URL':
                if re.search('[A-Z]', word[0]):
                    in_cap += 1
                if re.search('^[A-Z][A-Z]+$', word):
                    all_cap += 1
        if in_cap > max_in_cap:
            max_in_cap = in_cap
        if all_cap > max_all_cap:
            max_all_cap = all_cap
        cf.append(in_cap/15.0) # for 80/20: 29.0, 50/50: 15
        cf.append(all_cap/14.0) # for 80/20: 28.0, 50/50: 14
        cap_feats.append(cf)
    print max_in_cap, max_all_cap
    return np.array(cap_feats)

def intensifiers(tweets, size):
    """ get intensifier features """
    intense_words = ['amazingly', '-ass', 'astoundingly', 'awful', 'bare', 'bloody', 'crazy', 
    'dead', 'dreadfully', 'colosally', 'especially', 'exceptionally', 'excessively', 'excessively', 
    'extremely', 'extraordinarily', 'fantastically', 'frightfully', 'fucking', 'fully', 'hella', 
    'holy', 'incredibly', 'insanely', 'mightily', 'moderately', 'most', 'outrageously', 'phenomenally',
    'precious', 'quite', 'radically', 'rather', 'real', 'really', 'remarkably', 'right', 'sick', 'so', 
    'somewhat', 'strikingly', 'super', 'supemely', 'surpassingly', 'terribly', 'terrifically', 'too', 
    'totally', 'uncommonly', 'unusually', 'veritable', 'very', 'wicked']
    intense_feats = []
    for tweet in tweets:
        ints = 0
        for word in tweet.split():
            if word.lower() in intense_words:
                ints = 1
        intense_feats.append(ints)
    return np.array(intense_feats).reshape((size, 1))


def brown_features(tweets):
    """  """
    brown_tweets = []
    dic = json.load(open('new_brown_dict.json'))
    for tweet in tweets:
        brown_tweet = ""
        for word in tweet:#.split():
            item = word.strip(".,'?!")
            if item in dic.keys():
                brown_tweet += str(dic[item])+" "
        brown_tweets.append(brown_tweet)
    return brown_tweets

def make_pairwise_brown(conversations):
    """ construct pairwise Brown clusters"""

    pairwise_brown_list = []
    dic = json.load(open('new_brown_dict.json'))
    pairwise_brown_dict = {}
    for conv in conversations:
        pairs = []
        for word in conv[0].split():
            pair = []
            item = word.strip(".,'?!")
            if item in dic.keys():
                for word2 in conv[1].split():
                    item2 = word2.strip(".,'?!")
                    if item2 in dic.keys():
                        #pairwise_brown_list.append([dic[item], dic[item2]])
                        pairwise_brown_dict[str([dic[item], dic[item2]])] = 0
    #print pairwise_brown_list
    #print len(pairwise_brown_list)
    #with open('pairwise_brown_list_bal.txt', 'w') as outfile:
     #  outfile.write(str(pairwise_brown_list))
    #print pairwise_brown_dict
    #print len(pairwise_brown_dict)
    with open('pairwise_brown_dict_bal.json', 'w') as outfile:
        json.dump(pairwise_brown_dict, outfile)

def pairwise_brown(conversations):
    """ get pairwise Brown features """
    pb_feats = []
    dic = json.load(open('new_brown_dict.json'))
    #pb_clusts = eval(open('pairwise_brown_list_bal.txt').read())
    pb_clust_dict = json.load(open('pairwise_brown_dict_imb.json'))#change bal/imb

    # get brown pairs for each conversation
    for conv in conversations:
        pairs = []
        for word in conv[0].split():
            pair = []
            item = word.strip(".,'?!")
            if item in dic.keys():
                for word2 in conv[1].split():
                    item2 = word2.strip(".,'?!")
                    if item2 in dic.keys():
                        pair.append([dic[item], dic[item2]])
                        pairs.append(str(pair)[1:-1])
        pairwise_feats = []
        # get pairwise brown feat array for each conversation
        for clust_pair in pb_clust_dict.keys():
            if clust_pair in pairs:
                pairwise_feats.append(1)
            else:
                    pairwise_feats.append(0)
        pb_feats.append(pairwise_feats)
    return np.array(pb_feats)



def pos_features(tagged):
    """ """
    pos_feats = []
    tag_list = ['N', 'O', '^', 'S', 'Z', 'V', 'L', 'M', 'A', 'R', '!', 'D', 'P', '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G']
    content_tags = ['N', 'O', '^', 'S', 'Z', 'V', 'L', 'M', 'A', 'R',]
    #max_tag = 0
    #max_ratio = 0
    max_dense = 0
    for tweet in tagged:
        total_tags = 0.000000000000000000001
        feat_list = []
        content_ts = 0.0
        for tag in tag_list:
            ratio = []
            #if tweet.split().count(tag) > max_tag:
             #   max_tag = tweet.split().count(tag)
            total_tags += tweet.split().count(tag)
            feat_list.append(tweet.split().count(tag)/11.0) # normalized (35 is highest count 50/50)
        # calculate ratio of each tag:
        for tag in tag_list:
            #if tweet.split().count(tag)/total_tags > max_ratio:
             #   max_ratio = tweet.split().count(tag)/total_tags
            feat_list.append((tweet.split().count(tag)/total_tags)/0.714285714286) # 0.714285714286 50/50 
        # calculate lexical density feature
        for tag in content_tags:
            content_ts += tweet.split().count(tag)
        lex_dense = content_ts/total_tags
        #if lex_dense > max_dense:
         #   max_dense = lex_dense
        feat_list.append(lex_dense/0.857142857143) # norm: 50/50: 0.857142857143 0.909090909091
        pos_feats.append(feat_list)
    #print max_dense
    return np.array(pos_feats)

def profile_info(tweeters): #removed friends
    """  """
    # gender (as inferred by their first name, compared to trends in U.S. Social Security records), number of friends, followers and statuses, their duration on Twitter, the average number of posts per day, their timezone, and whether or not they are verified by Twitter (designating a kind of celebrity status)
    name_dict = json.load(open("all_names.json"))
    authors = []
    regex = re.compile('[^a-zA-Z]') #remove non-alpha characters
    months_dict = {'Jan':334, 'Feb':306, 'Mar':275, 'Apr':245, 'May':214, 'Jun':184, 'Jul':153, 'Aug':122, 'Sep':92, 'Oct':61, 'Nov':31, 'Dec':0}
    months_dict2 = {'Jan':0, 'Feb':31, 'Mar':59, 'Apr':90, 'May':120, 'Jun':151}
    time_zones = {1.0: ['Alaska', 'Arizona', 'Atlantic Time (Canada)', 'Central Time (US & Canada)', 'Eastern Time (US & Canada)', 'Hawaii', 'Indiana (East)', 'Mountain Time (US & Canada)', 'Newfoundland', 'Pacific Time (US & Canada)'], 0.9: ['Dublin', 'Edinburgh', 'London'], 0.8: ['Brisbane', 'Melbourne', 'Sydney', 'Wellington'], 0.7: ['Nairobi', 'Pretoria'], 0.6: ['Chennai', 'Islamabad', 'Karachi', 'Mumbai'], 0.5: ['Greenland', None, 'International Date Line West', 'New Caledonia'], 0.4: ['Bangkok', 'Beijing', 'Hong Kong', 'Kuala Lumpur', 'New Delhi', 'Seoul', 'Singapore', 'Taipei', 'Tokyo'], 0.3: ['Amsterdam', 'Athens', 'Bern','Berlin', 'Brussels', 'Bucharest', 'Copenhagen', 'Helsinki', 'Lisbon', 'Ljubljana', 'Madrid', 'Paris', 'Riga', 'Stockholm', 'Warsaw'], 0.2: ['Brasilia', 'Rio', 'Santiago', 'Tijuana', 'Quito'], 0.1: ['Casablanca', "Nuku'alofa", 'West Central Africa'], 0.0: ['Abu Dhabi', 'Baghdad', 'Jerusalem', 'Kabul', 'Kuwait', 'Riyadh']}
    #max_friends = 0.0
    #max_followers = 0.0
    #max_ppd = 0.0
    #max_days = 0.0
    #max_statuses = 0.0
    for tweeter in tweeters:
        author = []
        # gender:
        name = regex.sub('', tweeter[0])
        name = ''.join(''.join(s)[:2] for _, s in groupby(name)) # remove repetitious letters
        if name.lower().strip(".:!?,()*") in name_dict['male']:
            m_score = name_dict['male'][name.lower()]
        else:
            m_score = 0.0
        if name in name_dict['female']:
            f_score = name_dict['female'][name]
        else:
            f_score = 0.0
        if m_score > f_score:
            gender = 1.0
        elif m_score < f_score:
            gender = 0.0
        else: 
            gender = 0.5
        author.append(gender)
        # friends count
        #if tweeter[1] > max_friends:
         #   max_friends = tweeter[1]
        author.append(tweeter[1]/63236.0) # 63236.0 for both
        # followers count:
        #if tweeter[2] > max_followers:
         #   max_followers = tweeter[2]
        author.append(tweeter[2]/500569.0)  # 119101.0 for 50/50, 500569.0 for 80/20 119101
        # statuses count:
        #if tweeter[3] > max_statuses:
         #   max_statuses = tweeter[3]  
        author.append(tweeter[3]/387531.0) # 150254 for 50/50, 387531.0 for 80/20
        # length of time on twitter:
        days = 0
        year_created = int(str(tweeter[4])[-2:])
        days += (13-year_created)*365
        month_created = str(tweeter[4])[4:7]
        month = months_dict[month_created]
        days += month
        day_created = int(str(tweeter[4])[8:10])
        days += day_created
        tweet_month = str(tweeter[5])[:3]
        days += (months_dict2[tweet_month])
        author.append(days/2749.0) #  2749 for both, 
        #if max_days < days:
         #   max_days = days
        ppd = tweeter[3]/days
        author.append(ppd/276.610278373) # 129.817275748 for 50/50, 276.610278373
        #if ppd > max_ppd:
         #   max_ppd = ppd
        # timezone:
        zone = [key for key, value in time_zones.iteritems() if tweeter[6] in value]
        if len(zone) != 0:
            place = zone[0]
        else:
            place = 0.5
        author.append(place)
        # verified:
        verified = 0
        if tweeter[7] == True:
            verified = 1
        author.append(verified)
        authors.append(author)
    #print 'friends:', max_friends, 'followers:', max_followers, 'statuses:', max_statuses
    #print "days:", max_days, "ppd:", max_ppd
    return np.array(authors)

def communication_feats(tweets):
    """ """
    communication_feats = []
    previous = []
    #max_rank = 0
    max_pms = 0
    for conversation in tweets:
        comm_feats = []
        aud = conversation['aud']
        auth = conversation['auth']
        aud_name = '@'+aud['screen_name']
        pms =  float(auth['history'].strip(":.,?!'").split().count(aud_name))
        #if pms > max_pms:
         #   max_pms = pms
        previous.append(pms)
        comm_feats.append(pms/1474.0)# 50/50: 1474, 80/20: 2804
        mentions = []
        for word in auth['history'].strip(":.,?!'").split():
            if word[0] == '@':
                mentions.append(word)
        ranked_mentions = Counter(mentions).most_common(1500)
        try:
            rank = float(ranked_mentions.index([item for item in ranked_mentions if item[0] == aud_name][0]))
        except: 
            rank = 1016#1144
        #if rank > max_rank:
         #   max_rank = rank
        rank = rank/1016.0# 50/50: 1016, 80/20: 1144
        comm_feats.append(rank)
        #communication_feats.append(comm_feats)
        auth_name = '@'+str(auth['screen_name'])
        aud_pms =  aud['history'].strip(":.,?!'").split().count(auth_name)
        mutual = 0
        if aud_pms > 1:
            mutual = 1
        comm_feats.append(mutual)
        communication_feats.append(comm_feats)
    #print max_pms
    return np.array(communication_feats)



