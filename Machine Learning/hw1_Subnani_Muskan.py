
# coding: utf-8

# In[2]:


#Muskan Subnani
#V487T392
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

#1. Load the Boston housing dataset
housing = pd.read_csv('data/boston_housing.data',header=None,sep=' ')
# Success
#housingdata.info
dfhousing=pd.DataFrame(housing)
noofcolumns= len(dfhousing.columns)
#print(noofcolumns)
#2. Muliply the housing prices column with 1000
dfhousing[noofcolumns-1]*=1000
#check
#dfhousingdata

#3.  Using train test split function from scikitlearn randomly set 30% of the data to be test set and
#the remaining to be training set.

train_set, test_set = train_test_split(housing, test_size=0.3, random_state=42)
#Success check
#train_set.info()
#test_set.info()

#4.Separate out the features and labels (response variable y, i.e., median value of housing prices for
#training and test set.)
housing_train_set = train_set.drop(noofcolumns-1, axis=1)
housing_train_set_labels = train_set[noofcolumns-1].copy()
housing_test_set = test_set.drop(noofcolumns-1, axis=1)
housing_test_set_labels = test_set[noofcolumns-1].copy()
#Success Check
#housing_train_set.info()
#housing_train_set_labels
#housing_test_set.info()
#housing_test_set_labels

#5.StandardScaler for the training set using pipeline
#name features and labels from training and test sets as X train, y train, X test, y test
scaler_pipeline = Pipeline([
('std_scaler', StandardScaler()),
])

X_train=scaler_pipeline.fit_transform(housing_train_set)
Y_train=housing_train_set_labels
X_test=scaler_pipeline.fit_transform(housing_test_set)
Y_test=housing_test_set_labels

#Success check
#X_train.shape
#Y_train.shape
#X_test.shape
#Y_test.shape

#6. Training Models
#Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

#some_data = housing_train_set.iloc[:5]
#some_labels = Y_train.iloc[:5]
#some_data_prepared = scaler_pipeline.transform(some_data)
#print("Predictions:", lin_reg.predict(some_data_prepared))
#print("Labels:", list(some_labels))

# DecisionTreeRegressor Model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, Y_train)

#housing_predictions = tree_reg.predict(X_train)
#tree_mse = mean_squared_error(Y_train, housing_predictions)
#tree_rmse = np.sqrt(tree_mse)
#tree_rmse

#RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, Y_train)

#SupportVectorRegressor(SVR)
sv_reg=SVR(kernel='rbf',gamma=0.01)
sv_reg.fit(X_train, Y_train)

#7. 10 fold Cross Validation
#function to display scores for each cross validation
def display_scores(modelname,scores):
    print(modelname)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
# Linear Regression Validation
lin_scores = cross_val_score(lin_reg, X_train, Y_train,
 scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores("Linear Regression",lin_rmse_scores)

#DecisionTreeRegressor Validation
tree_scores = cross_val_score(tree_reg, X_train, Y_train,
 scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores("Decision Tree Regression",tree_rmse_scores)

#RandomForestRegressor Validation
forest_scores = cross_val_score(forest_reg, X_train, Y_train,
 scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores("Random Forest Regression",forest_rmse_scores)

#SupportVectorRegressor Validation
sv_scores = cross_val_score(sv_reg, X_train, Y_train,
 scoring="neg_mean_squared_error", cv=10)
sv_rmse_scores = np.sqrt(-sv_scores)
display_scores("Support Vector Regression",sv_rmse_scores)

#8.Fine-Tune RandomForestRegressor Model using Grid Search
param_grid = [
{'n_estimators': [10, 20, 30, 40], 'max_features': [ 4, 8]}
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=10,
scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
     print(np.sqrt(-mean_score), params)
print("Best Parameter: ")
print(grid_search.best_params_)

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final Prediction: ")
print(final_rmse)



























# In[16]:




