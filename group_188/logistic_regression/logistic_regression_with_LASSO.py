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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)

file = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/train.csv')

# We use regularization to select features during the fitting process.
X = file.drop(columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])
y = file['Diagnosis']

# logistic regression:
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'tol': [1e-4, 1e-3, 1e-5],
    'penalty': ['l1']
}

# Initialize logistic regression model
log_reg = LogisticRegression(max_iter=50000)

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# Perform the grid search to find the best model
grid_search.fit(X, y)

# Get the model with the best cross-validation score and print the best score
best_log_reg = grid_search.best_estimator_
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")
# Best cross-validation accuracy score: 0.8823

best_log_reg.fit(X, y)

joblib.dump(best_log_reg, 'best_logistic_regression_with_LASSO_model.joblib')
