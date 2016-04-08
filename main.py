# -*- coding: utf-8 -*-

import sys
import numpy as np
import logit
from texttable import Texttable

print "loading matrices"

""" load labels """
labels_bal = np.load('y_array_bal.npy')
labels_imb = np.load('y_array_imb.npy')

""" load feature matrices """
### TWEETS ###
# unigrams:
X_unigrams_bal = np.load('X_unigram_bal.npy')
X_unigrams_imb = np.load('X_unigram_imb.npy')
# bigrams:
X_bigrams_bal = np.load('X_bigram_bal.npy')
X_bigrams_imb = np.load('X_bigram_imb.npy')
# Brown unigrams:
X_brown_uni_bal = np.load('X_brown_uni_bal.npy')
X_brown_uni_imb = np.load('X_brown_uni_imb.npy')
# Brown unigrams:
X_brown_bi_bal = np.load('X_brown_bi_bal.npy')
X_brown_bi_imb = np.load('X_brown_bi_imb.npy')
# POS features:
X_pos_bal = np.load('X_pos_bal.npy')
X_pos_imb = np.load('X_pos_imb.npy')
# pron features:
X_pron_bal = np.load('X_pron_bal.npy')
X_pron_imb = np.load('X_pron_imb.npy')
# pron features:
X_cap_bal = np.load('X_cap_bal.npy')
X_cap_imb = np.load('X_cap_imb.npy')
# intensifiers:
X_intense_bal = np.load('X_intense_bal.npy')
X_intense_imb = np.load('X_intense_imb.npy')
# ALL TWEET FEATURES:
X_array1_bal = np.hstack((X_unigrams_bal, X_bigrams_bal))
X_array2_bal = np.hstack((X_array1_bal, X_brown_uni_bal))
X_array3_bal = np.hstack((X_array2_bal, X_brown_bi_bal))
X_array4_bal = np.hstack((X_array3_bal, X_pos_bal))
X_array5_bal = np.hstack((X_array4_bal, X_pron_bal))
X_array6_bal = np.hstack((X_array5_bal, X_cap_bal))
X_array7_bal = np.hstack((X_array6_bal, X_intense_bal))
X_array1_imb = np.hstack((X_unigrams_imb, X_bigrams_imb))
X_array2_imb = np.hstack((X_array1_imb, X_brown_uni_imb))
X_array3_imb = np.hstack((X_array2_imb, X_brown_bi_imb))
X_array4_imb = np.hstack((X_array3_imb, X_pos_imb))
X_array5_imb = np.hstack((X_array4_imb, X_pron_imb))
X_array6_imb = np.hstack((X_array5_imb, X_cap_imb))
X_array7_imb = np.hstack((X_array6_imb, X_intense_imb))

### AUTHOR ### 
# author profiles:
X_auth_profs_bal = np.load('X_auth_profs_bal.npy')
X_auth_profs_imb = np.load('X_auth_profs_imb.npy')
# author descriptions:
X_auth_descs_bal = np.load('X_auth_descs_bal.npy')
X_auth_descs_imb = np.load('X_auth_descs_imb.npy')
# TWEET + AUTHOR
X_array8_bal = np.hstack((X_array7_bal, X_auth_profs_bal))
X_array9_bal = np.hstack((X_array8_bal, X_auth_descs_bal))
X_array8_imb = np.hstack((X_array7_imb, X_auth_profs_imb))
X_array9_imb = np.hstack((X_array8_imb, X_auth_descs_imb))

### AUDIENCE ###
# audience descriptions:
X_aud_descs_bal = np.load('X_aud_descs_bal.npy')
X_aud_descs_imb = np.load('X_aud_descs_imb.npy')
# author profiles:
X_aud_profs_bal = np.load('X_aud_profs_bal.npy')
X_aud_profs_imb = np.load('X_aud_profs_imb.npy')
# TWEET + AUDIENCE
X_array10_bal = np.hstack((X_array7_bal, X_aud_profs_bal))
X_array11_bal = np.hstack((X_array10_bal, X_aud_descs_bal))
X_array10_imb = np.hstack((X_array7_imb, X_aud_profs_imb))
X_array11_imb = np.hstack((X_array10_imb, X_aud_descs_imb))

### ENVIRONMENT ###
# pairwise Brown:
X_pairwise_brown_bal = np.load('X_pairwise_brown_bal.npy')
X_pairwise_brown_imb = np.load('X_pairwise_brown_imb.npy')
# original unigrams:
X_orig_unigram_bal = np.load('X_orig_unigram_bal.npy')
X_orig_unigram_imb = np.load('X_orig_unigram_imb.npy')
# TWEET + ENVIRONMENT:
X_array12_bal = np.hstack((X_array7_bal, X_pairwise_brown_bal))
X_array13_bal = np.hstack((X_array12_bal, X_aud_descs_bal))
X_array12_imb = np.hstack((X_array7_imb, X_orig_unigram_imb))
X_array13_imb = np.hstack((X_array12_imb, X_orig_unigram_imb))

### ALL FEATS ###
X_array14_bal = np.hstack((X_array9_bal, X_array11_bal))
X_array15_bal = np.hstack((X_array14_bal, X_array13_bal))
X_array14_imb = np.hstack((X_array9_imb, X_array11_imb))
X_array15_imb = np.hstack((X_array14_imb, X_array13_imb))

def run_ahst(combination):
    """ Get keys and labels according to dataset size """
    """ make train / test splits """
    n_folds = 5
    fold_size_bal = len(labels_bal)/n_folds
    fold_size_imb = len(labels_imb)/n_folds

    split_start_bal = 0
    split_end_bal = fold_size_bal
    split_start_imb = 0
    split_end_imb = fold_size_imb

    acc_bal, prec_bal, rec_bal, f1_bal = 0, 0, 0, 0
    acc_imb, prec_imb, rec_imb, f1_imb = 0, 0, 0, 0

    """ import salient terms matrix """
    fold = 1
    for iteration in range(0, n_folds):
        print "running %s hst" % combination, fold
        # get y_array train / test splits:
        y_train_bal = np.hstack((labels_bal[:int(split_start_bal)], labels_bal[int(split_end_bal):]))
        y_test_bal = labels_bal[int(split_start_bal):int(split_end_bal)]

        y_train_imb = np.hstack((labels_imb[:int(split_start_imb)], labels_imb[int(split_end_imb):]))
        y_test_imb = labels_imb[int(split_start_imb):int(split_end_imb)]

        # author historical salient terms:
        X_ahst_bal = np.load('X_ahst%s_bal.npy' % iteration)
        X_ahst_imb = np.load('X_ahst%s_imb.npy' % iteration)

        if combination == 'tauth':
            X_ahst_bal = np.hstack((X_array9_bal, X_ahst_bal))
            X_ahst_imb = np.hstack((X_array9_imb, X_ahst_imb))

        # audience (response) historical salient terms:
        if combination == 'r' or combination == 'taud':
            X_ahst_bal = np.load('X_rhst%s_bal.npy' % iteration)
            X_ahst_imb = np.load('X_rhst%s_imb.npy' % iteration)

        if combination == 'taud':
            X_rhst_bal = np.load('X_rhst%s_bal.npy' % iteration)
            X_rhst_imb = np.load('X_rhst%s_imb.npy' % iteration)
            X_ahst_bal = np.hstack((X_array11_bal, X_rhst_bal))
            X_ahst_imb = np.hstack((X_array11_imb, X_rhst_imb))

        if combination == 'all':
            X_rhst_bal = np.load('X_rhst%s_bal.npy' % iteration)
            X_rhst_imb = np.load('X_rhst%s_imb.npy' % iteration)            
            X_rhst_bal = np.hstack((X_array15_bal, X_rhst_bal))
            X_rhst_imb = np.hstack((X_array15_imb, X_rhst_imb))
            X_ahst_bal = np.hstack((X_rhst_bal, X_ahst_bal))
            X_ahst_imb = np.hstack((X_rhst_imb, X_ahst_imb))

        # get X array train/test splits:
        X_train_bal = np.vstack((X_ahst_bal[:int(split_start_bal),:], X_ahst_bal[int(split_end_bal):,:]))
        X_test_bal = X_ahst_bal[int(split_start_bal):int(split_end_bal)]

        X_train_imb = np.vstack((X_ahst_imb[:int(split_start_imb),:], X_ahst_imb[int(split_end_imb):,:]))
        X_test_imb = X_ahst_imb[int(split_start_imb):int(split_end_imb)]

        # run logistic regression:
        test_bal = logit.run_logit(X_train_bal, X_test_bal, y_train_bal, y_test_bal)
        acc_bal += test_bal[0]
        prec_bal += test_bal[1]
        rec_bal += test_bal[2]
        f1_bal += test_bal[3]

        test_imb = logit.run_logit(X_train_imb, X_test_imb, y_train_imb, y_test_imb)
        acc_imb += test_imb[0]
        prec_imb += test_imb[1]
        rec_imb += test_imb[2]
        f1_imb += test_imb[3]

        # update to next interation:
        split_start_bal += fold_size_bal
        split_end_bal += fold_size_bal

        split_start_imb += fold_size_imb
        split_end_imb += fold_size_imb

        #update fold no:
        fold += 1

    return acc_bal/5, prec_bal/5, rec_bal/5, f1_bal/5, acc_imb/5, prec_imb/5, rec_imb/5, f1_imb/5

def run_experi(X_matrix_bal, X_matrix_imb, feat_name):
    """ Get keys and labels according to dataset size """
    """ make train / test splits """
    n_folds = 5
    fold_size_bal = len(labels_bal)/n_folds
    fold_size_imb = len(labels_imb)/n_folds


    split_start_bal = 0
    split_end_bal = fold_size_bal
    split_start_imb = 0
    split_end_imb = fold_size_imb

    acc_bal, prec_bal, rec_bal, f1_bal = 0, 0, 0, 0
    acc_imb, prec_imb, rec_imb, f1_imb = 0, 0, 0, 0

    """ import salient terms matrix """
    fold = 1
    for iteration in range(0, n_folds):
        print "running %s fold" % feat_name, fold
        # get y_array train / test splits:
        y_train_bal = np.hstack((labels_bal[:int(split_start_bal)], labels_bal[int(split_end_bal):]))
        y_test_bal = labels_bal[int(split_start_bal):int(split_end_bal)]

        y_train_imb = np.hstack((labels_imb[:int(split_start_imb)], labels_imb[int(split_end_imb):]))
        y_test_imb = labels_imb[int(split_start_imb):int(split_end_imb)]

        # get X array train/test splits:
        X_train_bal = np.vstack((X_matrix_bal[:int(split_start_bal),:], X_matrix_bal[int(split_end_bal):,:]))
        X_test_bal = X_matrix_bal[int(split_start_bal):int(split_end_bal)]

        X_train_imb = np.vstack((X_matrix_imb[:int(split_start_imb),:], X_matrix_imb[int(split_end_imb):,:]))
        X_test_imb = X_matrix_imb[int(split_start_imb):int(split_end_imb)]

        # run logistic regression:
        test_bal = logit.run_logit(X_train_bal, X_test_bal, y_train_bal, y_test_bal)
        acc_bal += test_bal[0]
        prec_bal += test_bal[1]
        rec_bal += test_bal[2]
        f1_bal += test_bal[3]

        test_imb = logit.run_logit(X_train_imb, X_test_imb, y_train_imb, y_test_imb)
        acc_imb += test_imb[0]
        prec_imb += test_imb[1]
        rec_imb += test_imb[2]
        f1_imb += test_imb[3]

        # update to next interation:
        split_start_bal += fold_size_bal
        split_end_bal += fold_size_bal

        split_start_imb += fold_size_imb
        split_end_imb += fold_size_imb

        #update fold no:
        fold += 1

    return acc_bal/5, prec_bal/5, rec_bal/5, f1_bal/5, acc_imb/5, prec_imb/5, rec_imb/5, f1_imb/5

""" run experiments """
# TWEET FEATS:
unigram_result = run_experi(X_unigrams_bal, X_unigrams_imb, "unigrams")
bigram_result = run_experi(X_bigrams_bal, X_bigrams_imb, "bigrams")
brown_uni_result = run_experi(X_brown_uni_bal, X_brown_uni_imb, "Brown unigrams")
brown_bi_result = run_experi(X_brown_bi_bal, X_brown_bi_imb, "Brown bigrams")
pos_result = run_experi(X_pos_bal, X_pos_imb, "POS features")
pron_result = run_experi(X_pron_bal, X_pron_imb, "POS features")
cap_result = run_experi(X_cap_bal, X_cap_imb, "capital features")
intense_result = run_experi(X_intense_bal, X_intense_imb, "intensifiers")
tweet_result = run_experi(X_array7_bal, X_array7_imb, "tweet feats")
# AUTHOR FEATS
ahst_result = run_ahst('no')
auth_profs_result = run_experi(X_auth_profs_bal, X_auth_profs_imb, "author profs")
auth_descs_result = run_experi(X_auth_descs_bal, X_auth_descs_imb, "author profs")
tweet_auth_result = run_ahst('tauth')
# AUDIENCE FEATS
rhst_result = run_ahst('r')
aud_profs_result = run_experi(X_aud_profs_bal, X_aud_profs_imb, "audience profs")
aud_descs_result = run_experi(X_auth_descs_bal, X_auth_descs_imb, "author descs")
tweet_aud_result = run_ahst('taud')
# ENVIRONMENT FEATS
pairwise_brown_result = run_experi(X_pairwise_brown_bal, X_pairwise_brown_imb, "pairwise Brown")
orig_unigram_result = run_experi(X_orig_unigram_bal, X_orig_unigram_imb, "original unigrams")
tweet_env_result = run_experi(X_array13_bal, X_array13_imb, "TWEET + ENVIRONMENT")
# ALL FEATS
#all_feats_result = run_experi(X_array15_bal, X_array15_imb, "ALL FEATS")
all_feats_result = run_ahst('all')

""" Create results table """
results = Texttable()
results.add_row([' ', 'Accuracy', 'Precision', 'Recall', 'F1', 'Accuracy', 'Precision', 'Recall', 'F1'])
results.add_row(['Unigrams', unigram_result[0], unigram_result[1], unigram_result[2], unigram_result[3], unigram_result[4], unigram_result[5], unigram_result[6], unigram_result[7]])
results.add_row(['Bigrams', bigram_result[0], bigram_result[1], bigram_result[2], bigram_result[3], bigram_result[4], bigram_result[5], bigram_result[6], bigram_result[7]])
results.add_row(['Brown unigrams', brown_uni_result[0], brown_uni_result[1], brown_uni_result[2], brown_uni_result[3], brown_uni_result[4], brown_uni_result[5], brown_uni_result[6], brown_uni_result[7]])
results.add_row(['Brown bigrams', brown_bi_result[0], brown_bi_result[1], brown_bi_result[2], brown_bi_result[3], brown_bi_result[4], brown_bi_result[5], brown_bi_result[6], brown_bi_result[7]])
results.add_row(['POS feats', pos_result[0], pos_result[1], pos_result[2], pos_result[3], pos_result[4], pos_result[5], pos_result[6], pos_result[7]])
results.add_row(['Pron feats', pron_result[0], pron_result[1], pron_result[2], pron_result[3], pron_result[4], pron_result[5], pron_result[6], pron_result[7]])
results.add_row(['Capital feats', cap_result[0], cap_result[1], cap_result[2], cap_result[3], cap_result[4], cap_result[5], cap_result[6], cap_result[7]])
results.add_row(['Intense feats', intense_result[0], intense_result[1], intense_result[2], intense_result[3], intense_result[4], intense_result[5], intense_result[6], intense_result[7]])
results.add_row(['TWEET FEATS', tweet_result[0], tweet_result[1], tweet_result[2], tweet_result[3], tweet_result[4], tweet_result[5], tweet_result[6], tweet_result[7]])
results.add_row(['Auth hist salient', ahst_result[0], ahst_result[1], ahst_result[2], ahst_result[3], ahst_result[4], ahst_result[5], ahst_result[6], ahst_result[7]])
results.add_row(['Author profiles', auth_profs_result[0], auth_profs_result[1], auth_profs_result[2], auth_profs_result[3], auth_profs_result[4], auth_profs_result[5], auth_profs_result[6], auth_profs_result[7]])
results.add_row(['Author descriptions', auth_descs_result[0], auth_descs_result[1], auth_descs_result[2], auth_descs_result[3], auth_descs_result[4], auth_descs_result[5], auth_descs_result[6], auth_descs_result[7]])
results.add_row(['TWEET + AUTHOR', tweet_auth_result[0], tweet_auth_result[1], tweet_auth_result[2], tweet_auth_result[3], tweet_auth_result[4], tweet_auth_result[5], tweet_auth_result[6], tweet_auth_result[7]])
results.add_row(['Aud hist salient', rhst_result[0], rhst_result[1], rhst_result[2], rhst_result[3], rhst_result[4], rhst_result[5], rhst_result[6], rhst_result[7]])
results.add_row(['Audience descriptions', aud_descs_result[0], aud_descs_result[1], aud_descs_result[2], aud_descs_result[3], aud_descs_result[4], aud_descs_result[5], aud_descs_result[6], aud_descs_result[7]])
results.add_row(['Audience profiles', aud_profs_result[0], aud_profs_result[1], aud_profs_result[2], aud_profs_result[3], aud_profs_result[4], aud_profs_result[5], aud_profs_result[6], aud_profs_result[7]])
results.add_row(['TWEET + AUDIENCE', tweet_aud_result[0], tweet_aud_result[1], tweet_aud_result[2], tweet_aud_result[3], tweet_aud_result[4], tweet_aud_result[5], tweet_aud_result[6], tweet_aud_result[7]])
results.add_row(['Pairwise Brown', pairwise_brown_result[0], pairwise_brown_result[1], pairwise_brown_result[2], pairwise_brown_result[3], pairwise_brown_result[4], pairwise_brown_result[5], pairwise_brown_result[6], pairwise_brown_result[7]])
results.add_row(['Unigrams', orig_unigram_result[0], orig_unigram_result[1], orig_unigram_result[2], orig_unigram_result[3], orig_unigram_result[4], orig_unigram_result[5], orig_unigram_result[6], orig_unigram_result[7]])
results.add_row(['TWEET + ENVIRONMENT', tweet_env_result[0], tweet_env_result[1], tweet_env_result[2], tweet_env_result[3], tweet_env_result[4], tweet_env_result[5], tweet_env_result[6], tweet_env_result[7]])
results.add_row(['TWEET + AUTH + AUD + ENVIRONMENT', all_feats_result[0], all_feats_result[1], all_feats_result[2], all_feats_result[3], all_feats_result[4], all_feats_result[5], all_feats_result[6], all_feats_result[7]])
print results.draw()
