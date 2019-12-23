#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

X, y = load_digits(return_X_y=True)
X_train, X_test, Y_traintemp, Y_testtemp = train_test_split(X,y, test_size=0.4, random_state=43)

# True for all 8s and 5s, False for all other digits.
Y_traintempnp= np.array(Y_traintemp)
Y_testtempnp= np.array(Y_testtemp)

Y_train = ((Y_traintempnp == 8) | (Y_traintempnp==5))
Y_test = ((Y_testtempnp == 8) | (Y_testtempnp==5))

#Pipeline for missing values-Imputer, standard scaler

imputerscaler_pipeline = Pipeline([
('imputer', Imputer(strategy="median")),   
('std_scaler', StandardScaler()),
])

# apply pipeline to  training and test data
X_train_tr=imputerscaler_pipeline.fit_transform(X_train)
X_test_tr=imputerscaler_pipeline.fit_transform(X_test)

#create instance of svm
svmclf = svm.SVC()

#8.Fine-Tune svmclf Model using Grid Search
param_grid = [
  { 'kernel': ['linear'], 'C': [0.01,0.1,1,10,100] },
  {'kernel': ['rbf'], 'gamma': ['auto'], 'C': [0.01,0.1,1,10,100] },
    {'kernel': ['poly'], 'gamma': ['auto'], 'degree' : [2,4,6], 'C': [0.01,0.1,1,10,100] }  
 ]

grid_search = GridSearchCV(svmclf, param_grid, cv=3,
scoring='roc_auc')

grid_search.fit(X_train_tr, Y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(mean_score, params)
print("Best Parameter: ")
print(grid_search.best_params_)

final_model = grid_search.best_estimator_
final_model_train = final_model.fit(X_train_tr, Y_train)

final_predictions = final_model.predict(X_test_tr)
final_auc = roc_auc_score(Y_test, final_predictions)
print("Final Test AUC Score: ")
print(final_auc)


# In[ ]:




