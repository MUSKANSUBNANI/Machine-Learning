import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

#1. Load the sign language dataset
# First row is the header                       
sign_lang_train_data=pd.read_csv('sign_mnist_train.csv', header=[0])
sign_lang_test_data=pd.read_csv("sign_mnist_test.csv", header=[0])

X_train= sign_lang_train_data.drop('label', axis=1)
Y_train= [int(numeric_string) for numeric_string in sign_lang_train_data["label"]]
Y_train= np.array(Y_train)
Y_train_bin= label_binarize(Y_train, classes=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
#X_train.shape #(27455, 784)

X_test= sign_lang_test_data.drop('label', axis=1)
Y_test= [int(numeric_string) for numeric_string in sign_lang_test_data["label"]]
Y_test= np.array(Y_test)
Y_test_bin= label_binarize(Y_test, classes=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
Y_test.shape #(7172,)

n_classes = 24

#Pipeline for missing values-Imputer , standard scaler
#imputerscaler_pipeline = Pipeline([
#('imputer', Imputer(strategy="median")),   
#('std_scaler', StandardScaler()),
#])
# apply pipeline to  traing and test data
#X_train_tr=imputerscaler_pipeline.fit_transform(X_train)
#X_test_tr=imputerscaler_pipeline.fit_transform(X_test)
clf = MultinomialNB()
y_score= cross_val_predict(clf, X_train, Y_train, cv=3)
print (accuracy_score(Y_train, y_score)*100)
# Cross validate the scores
print (classification_report(Y_train, y_score, labels=range(0,25)))

conf_mx = confusion_matrix(Y_train, y_score)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

#Multi-label output to create P-R and roc curve
ovrclassifier = OneVsRestClassifier(clf)
y_score_bin = cross_val_predict(ovrclassifier, X_train, Y_train_bin, cv=3, method="predict_proba")

precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_train_bin[:, i],
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

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_train_bin[:, i],
                                  y_score_bin[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()


# Passing Test Data to Model
clf.fit(X_train, Y_train)
test_score=clf.predict(X_test) 
print("TEST DATA ACCURACY SCORE AND CLASSIFICATION REPORT")
print (accuracy_score(Y_test, test_score)*100)
# Cross validate the scores
print (classification_report(Y_test, test_score, labels=range(0,25)))
