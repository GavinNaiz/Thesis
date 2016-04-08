# -*- coding: utf-8 -*-

from __future__ import division
import sys
import numpy as np
import json
import ahst, logit


""" Get keys and labels according to dataset size """
if sys.argv[1] == '50':
    keys = np.load('keys_bal.npy')
    labels = np.load('y_array_bal.npy')
elif sys.argv[1] == '80':
    keys = np.load('keys_imb.npy')
    labels = np.load('y_array_imb.npy')


""" make train / test splits """
n_folds = 5
fold_size = len(keys)/n_folds

split_start = 0
split_end = fold_size

"""acc = 0
prec = 0
rec = 0
f1 = 0"""

salient_dict = {}
for iteration in range(0, n_folds):
    # get y_array train / test splits:
    y_train = np.hstack((labels[:int(split_start)], labels[int(split_end):]))
    y_test = labels[int(split_start):int(split_end)]

    # get train keys:
    keys_train = np.hstack((keys[:int(split_start)], keys[int(split_end):]))

    # run author historical salient terms to get feature matrix:
    salient_matrix = ahst.get_salience_matrix(keys, ahst.get_salient_set(keys_train))
    salient_dict[iteration] = salient_matrix
    
    # get X array train/test splits:
    #X_train = np.vstack((salient_matrix[:int(split_start),:], salient_matrix[int(split_end):,:]))
    #X_test = salient_matrix[int(split_start):int(split_end)]

    # run logistic regression:
    #logistic = logit.run_logit(X_train, X_test, y_train, y_test)
    #print logistic


    # update to next interation:
    split_start += fold_size
    split_end += fold_size

for key, val in salient_dict.iteritems():
    np.save('X_rhst%s_imb.npy' % key, val)

