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

# We checked the scatter plot and found that the correlations between the predictors
# , and we decided to drop the numerical variables with low correlation with the results.

X = file.drop(columns=['PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                     'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diagnosis',
                     'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                     'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])


# Below is a 7-variable version
# X = file.drop(columns=['PatientID', 'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
#                        'Smoking', 'AlcoholConsumption', 'PhysicalActivity','DietQuality',
#                        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Depression',
#                        'HeadInjury', 'Hypertension', 'Diagnosis', 'SystolicBP', 'DiastolicBP',
#                        'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
#                        'CholesterolTriglycerides', 'MemoryComplaints', 'Confusion',
#                        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
#                        'Forgetfulness', 'DoctorInCharge'])

y = file['Diagnosis']

# Random forest:
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
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=25, scoring='accuracy', n_jobs=-1)

# Perform the grid search to find the best model
grid_search.fit(X, y)

# Get the model with the best cross-validation score and print the best score
best_rf = grid_search.best_estimator_
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")
# For 10-folds CV
# Best cross-validation accuracy score: 0.9555

# For 20-folds CV
# Best cross-validation accuracy score: 0.9568
best_rf.fit(X, y)


joblib.dump(best_rf, 'best_random_forest_model.joblib')

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

pd.set_option('display.max_columns', None)

# Load dataset
file = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/train.csv')

# Drop low-correlation numerical variables
X = file.drop(columns=['PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                       'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diagnosis',
                       'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                       'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])

y = file['Diagnosis']

# Random forest hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize the Random Forest model
rf = RandomForestClassifier()

# Set up GridSearchCV with 25-fold cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=25, scoring='accuracy', n_jobs=-1)

# Perform the grid search
grid_search.fit(X, y)

# Get the best model from the grid search
best_rf = grid_search.best_estimator_
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

# Fit the best model on the training data
best_rf.fit(X, y)

# Save the model to a file
joblib.dump(best_rf, 'best_random_forest_model.joblib')
