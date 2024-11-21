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
# which are Diagnosis and DectorInCharge and the results are near zero. So we decided to drop them.
X = file.drop(columns=['PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                     'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diagnosis',
                     'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                     'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])
y = file['Diagnosis']

# vector support machine:
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Initialize the SVM model
svm = SVC()

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)

# Perform the grid search to find the best model
grid_search.fit(X, y)

# Get the model with the best cross-validation score and print the best score
best_svm = grid_search.best_estimator_
print(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")
# Best cross-validation accuracy score: 0.8418

best_svm.fit(X, y)

joblib.dump(best_svm, 'best_svm_model.joblib')
