#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, cache=True)
#mnist
Y= [int(numeric_string) for numeric_string in mnist["target"]]
Y= np.array(Y)
#Y.shape
X=mnist["data"]

X_train, X_test, Y_traintemp, Y_testtemp = train_test_split(X,Y, test_size=0.2, random_state=43)

# True for all 8s, False for all other digits.
Y_train = (Y_traintemp == 8)
Y_test = (Y_testtemp == 8)

#Pipeline for missing values-Imputer , standard scaler
imputerscaler_pipeline = Pipeline([
('imputer', Imputer(strategy="median")),   
('std_scaler', StandardScaler()),
])
# apply pipeline to  traing and test data
X_train_tr=imputerscaler_pipeline.fit_transform(X_train)
X_test_tr=imputerscaler_pipeline.fit_transform(X_test)

#Training Logistic Regression
logreg= LogisticRegression(solver="liblinear")
# Prediction on training data with 3 fold cross validation
y_train_pred= cross_val_predict(logreg, X_train_tr, Y_train, cv=3)

#Compute andPrint Confusion matrix, recall and predict
print( "Confusion Matrix: ")
print (confusion_matrix(Y_train, y_train_pred))
print("Precision Score: ")
print(precision_score(Y_train, y_train_pred))
print("Recall Score: ")
print(recall_score(Y_train, y_train_pred))

# compute probabilty using predict_proba method
y_probas = cross_val_predict(logreg, X_train_tr, Y_train, cv=3, method="predict_proba")
 # score = proba of positive class 
y_scores= y_probas[:, 1]

# Now compute the data needed for P-R curve
precisions, recalls, thresholds = precision_recall_curve(Y_train, y_scores)

# simple function to create the PR curve 
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

# call the function to plot the P-R curve
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#by looking at the P-R curve Threshold 0.8 give more than 90% precision
y_train_pred_90 = (y_scores > 0.75)

print(" Precision and Recall scores for TRAIN DATA ")

print (precision_score(Y_train, y_train_pred_90))

print (recall_score(Y_train, y_train_pred_90))

logreg.fit(X_train_tr, Y_train)

test_probas=logreg.predict_proba(X_test_tr) 

test_score= test_probas[:, 1]

y_test_pred_90 = (test_score > 0.75)

print(" Precision and Recall scores for TEST DATA ")

print (precision_score(Y_test, y_test_pred_90))

print (recall_score(Y_test, y_test_pred_90))












# In[ ]:




