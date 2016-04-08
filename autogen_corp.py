# -*- coding: utf-8 -*-

import tweepy
import json



auth_pos = json.load(open('hash_pos_new.json'))

orig_pos1 = json.load(open('origs1_hash_pos.json'))
orig_pos2 = json.load(open('origs2_hash_pos.json'))
orig_pos3 = json.load(open('origs3_hash_pos.json'))

orig_pos_dict = {}

for k, v in orig_pos1.iteritems():
    orig_pos_dict[k] = v
print len(orig_pos_dict)

for k, v in orig_pos2.iteritems():
    orig_pos_dict[k] = v
print len(orig_pos_dict)

for k, v in orig_pos3.iteritems():
    orig_pos_dict[k] = v
print len(orig_pos_dict)

hash_dict = {}

for key, val in auth_pos.iteritems():
    hash_dict[key] = {'author': val}
    hash_dict[key]['label'] = 1

for key, val in hash_dict.iteritems():
    print key, val
    print " "