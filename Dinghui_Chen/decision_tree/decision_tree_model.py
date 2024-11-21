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

file = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/train.csv')

# We checked the scatter plot and found that the correlations between the predictors
# which are Diagnosis and DectorInCharge and the results are near zero. So we decided to drop them.
X = file.drop(columns=['PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                     'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diagnosis',
                     'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                     'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])
y = file['Diagnosis']

# decision tree：
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 15],
    'criterion': ['gini', 'entropy']
}

# Initialize the Decision Tree model
tree = DecisionTreeClassifier()

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# Perform the grid search to find the best model
grid_search.fit(X, y)

# Get the model with the best cross-validation score and print the best score
best_tree = grid_search.best_estimator_
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")
# Best cross-validation accuracy score: 0.9521
best_tree.fit(X, y)

joblib.dump(best_tree, 'best_decision_tree_model.joblib')
