#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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

#instance of RandomForestClassifier
rfclf = RandomForestClassifier()

param_grid = { 
    'n_estimators': [100, 200, 300, 400]
}

grid_search = GridSearchCV(rfclf, param_grid, cv=3,
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
#instance of PCA
pca = PCA()
X_train_reduced = pca.fit_transform(X_train_tr)

#empty lists
d_list=[]
test_auc_list=[]

# for loop
for d in range(2, 64+1, 2):
    grid_search.fit(X_train_reduced[:, 0:d], Y_train)
    print("Projection to dimension " + str(d) + ": " +str(grid_search.best_params_))
    final_model_reduced=grid_search.best_estimator_
    X_test_reduced=np.dot(X_test_tr, pca.components_.T[:,0:d])
    final_predictions_reduced= final_model_reduced.predict(X_test_reduced)
    final_auc_reduced = roc_auc_score(Y_test, final_predictions_reduced)
    print("Projection to dimension " + str(d) + ": TEST AUC Score " +str(final_auc_reduced))
    d_list.append(d)
    test_auc_list.append(final_auc_reduced)

    #plot the graph
plt.plot(d_list, test_auc_list )
plt.xlabel("Reduced Dimension")
plt.ylabel("Test ROC")
plt.show()
    


# In[ ]:




