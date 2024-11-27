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

# We use Gini importance to select features during the fitting process.

X = file.drop(columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])


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
grid_search_all = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search_all.fit(X, y)

# Best model after the first GridSearchCV
best_rf_all = grid_search_all.best_estimator_

# Display the best cross-validation score and parameters
print(f"Best cross-validation accuracy with all features: {grid_search_all.best_score_:.4f}")
print(f"Best parameters with all features: {grid_search_all.best_params_}")

# Step 2: Extract Gini importance and select important features
importances = best_rf_all.feature_importances_
feature_names = X.columns

# Create a DataFrame to store feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance (Gini):")
print(importance_df)

# Step 3: Perform GridSearchCV again on selected top 17 features as the score dropped rapidly
# after the feature with 17th highest score
top_features = importance_df.head(17)['Feature']
X_selected = X[top_features]

grid_search_selected = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search_selected.fit(X_selected, y)

best_rf_selected = grid_search_selected.best_estimator_

print(f"\nBest cross-validation accuracy with selected features: {grid_search_selected.best_score_:.4f}")
print(f"Best parameters with selected features: {grid_search_selected.best_params_}")

joblib.dump(grid_search_selected, 'best_random_forest_model_Gini_importance.joblib')
