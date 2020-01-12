#!/usr/bin/env python
# coding: utf-8

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# To plot figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import pandas as pd
#load training and test data function
def load_sign_training_data():
    csv_training_name = 'sign_mnist_train.csv'
    return pd.read_csv(csv_training_name, header=0)

def load_sign_test_data():
    csv_training_name = 'sign_mnist_test.csv'
    return pd.read_csv(csv_training_name, header=0)

#load training data
training_sign = load_sign_training_data()
training_sign.head()


#load test dta
test_sign = load_sign_test_data()
test_sign.head()

#making x and y for training
X_train = df = training_sign.loc[:, training_sign.columns != 'label']
y_train= [int(numeric_string) for numeric_string in training_sign["label"]]
y_train= np.array(y_train)
y_train_bin= label_binarize(y_train, classes=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

#making x and y for testing
X_test= test_sign.drop('label', axis=1)
y_test= [int(numeric_string) for numeric_string in test_sign["label"]]
y_test= np.array(y_test)
y_test_bin= label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

#number of letter we will use
n_classes = 24


#scale time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
full_pipeline = Pipeline([      
        ('std_scaler', StandardScaler()),
    ])

#scaler = StandardScaler()
X_train_tr = full_pipeline.fit_transform(X_train.astype(np.float64))
X_test_tr = full_pipeline.fit_transform(X_test.astype(np.float64))

# Fitting Random Forest Classification to the Training set
rfc_cls = RandomForestClassifier(n_estimators = 10, random_state = 42)

#for getting y_train_pred for matrix
y_train_pred = cross_val_predict(rfc_cls, X_train_tr, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)

#show confused matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

#show confused matrix
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

rfc_classifier = OneVsRestClassifier(rfc_cls)
y_score_bin = cross_val_predict(rfc_classifier, X_train_tr, y_train_bin, cv=3, method="predict_proba")
#show precision vs recall curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_train_bin[:, i],
                                                        y_score_bin[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

# roc curve
fpr = dict()
tpr = dict()
#show true positive and false positive rate
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_train_bin[:, i],
                                  y_score_bin[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()

# Passing Test Data to Model
rfc_cls.fit(X_train, y_train)
test_score=rfc_cls.predict(X_test) 
print("TEST DATA ACCURACY SCORE AND CLASSIFICATION REPORT")
print (accuracy_score(y_test, test_score)*100)
# Cross validate the scores
print (classification_report(y_test, test_score, labels=range(0,25)))

#now tuning for final model
from sklearn.model_selection import GridSearchCV

param_grid = [
   {'n_estimators': [10,20,30,40], 'max_features': [4,8]},
     ]

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(rfc_cls, param_grid, cv=3,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)

#find best params
grid_search.best_params_

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

final_model_rfc = grid_search.best_estimator_

final_rfc_classifier = OneVsRestClassifier(final_model_rfc)
y_score_bin_final = cross_val_predict(final_rfc_classifier, X_train_tr, y_train_bin, cv=3, method="predict_proba")

#show precision vs recall curce using best estimator
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_train_bin[:, i],
                                                        y_score_bin_final[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

# roc curve
fpr = dict()
tpr = dict()

#show true positive rate  vs false positive rate curce using best estimator

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_train_bin[:, i],
                                  y_score_bin_final[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()

# Passing Test Data to FINAL model
final_model_rfc.fit(X_train, y_train)
test_score=final_model_rfc.predict(X_test) 
print("TEST DATA ACCURACY SCORE AND CLASSIFICATION REPORT from Final Model")
print (accuracy_score(y_test, test_score)*100)
# Cross validate the scores
print (classification_report(y_test, test_score, labels=range(0,25)))





