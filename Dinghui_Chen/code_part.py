import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)

df = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/train.csv')

X = df.drop(columns=['Diagnosis', 'DoctorInCharge'])
y = df['Diagnosis']

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# ### The first model is logistic regression:
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'solver': ['lbfgs', 'saga'],
#     'tol': [1e-4, 1e-3, 1e-5]
# }
#
# # Initialize logistic regression model
# log_reg = LogisticRegression()
#
# # Set up GridSearchCV with 10-fold cross-validation
# grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
#
# # Perform the grid search to find the best model
# grid_search.fit(X, y)
#
# # Get the model with the best cross-validation score
# best_log_reg = grid_search.best_estimator_
#
# best_log_reg.fit(X, y)
#
# joblib.dump(best_log_reg, 'best_logistic_regression_model.joblib')
#
# ### The second model is vector support machine:
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': ['scale', 'auto']
# }
#
# # Initialize the SVM model
# svm = SVC()
#
# # Set up GridSearchCV with 10-fold cross-validation
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
#
# # Perform the grid search to find the best model
# grid_search.fit(X, y)
#
# # Get the model with the best cross-validation score
# best_svm = grid_search.best_estimator_
#
# best_svm.fit(X, y)
#
# joblib.dump(best_svm, 'best_svm_model.joblib')
#
# ### The 3rd model is decision tree：
# param_grid = {
#     'max_depth': [None, 10, 20, 30, 40],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 5, 10, 15],
#     'criterion': ['gini', 'entropy']
# }
#
# # Initialize the Decision Tree model
# tree = DecisionTreeClassifier()
#
# # Set up GridSearchCV with 10-fold cross-validation
# grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
#
# # Perform the grid search to find the best model
# grid_search.fit(X, y)
#
# # Get the model with the best cross-validation score
# best_tree = grid_search.best_estimator_
#
# best_tree.fit(X, y)
#
# joblib.dump(best_tree, 'best_decision_tree_model.joblib')

### The 4th model is random forest:
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize the Random Forest model
rf = RandomForestClassifier()

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# Perform the grid search to find the best model
grid_search.fit(X, y)

# Get the model with the best cross-validation score
best_rf = grid_search.best_estimator_

best_rf.fit(X, y)


joblib.dump(best_rf, 'best_random_forest_model.joblib')
