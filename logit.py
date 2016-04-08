# -*- coding: utf-8 -*-

from sklearn import linear_model, metrics

def run_logit(X_train, X_test, y_train, y_test):
    """ run logistic regression on salient terms matrix """
    logistic = linear_model.LogisticRegression(class_weight="balanced")
    acc = logistic.fit(X_train, y_train).score(X_test, y_test)
    pred = logistic.predict(X_test)
    f1 = metrics.f1_score(y_test, pred)
    prec = metrics.precision_score(y_test, pred)
    rec = metrics.recall_score(y_test, pred)
    return acc, prec, rec, f1