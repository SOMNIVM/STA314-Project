# Packages installation
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(context='notebook', palette='muted', style='darkgrid')

# import dataset
df = pd.read_csv('~/PycharmProjects/STA314-Project/train.csv')
print(df.head())

# #EDA
# ## Age
# plt.figure(figsize=(8, 5))
# sns.histplot(df['Age'], bins=15, kde=True)
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()
#
# ##Ethnicity Distribution
# plt.figure(figsize=(8, 5))
# sns.countplot(x='Ethnicity', data=df)
# plt.title('Ethnicity Distribution')
# plt.show()
#
# ##Functional Assessment by Alzheimer's Diagnosis
# plt.figure(figsize=(8, 5))
# sns.boxplot(x='Diagnosis', y='FunctionalAssessment', data=df)
# plt.title('Functional Assessment by Alzheimer\'s Diagnosis')
# plt.show()
#
# symptoms = ['Confusion', 'Disorientation', 'PersonalityChanges',
#             'DifficultyCompletingTasks', 'Forgetfulness']
#
# plt.figure(figsize=(12, 8))
# for i, symptom in enumerate(symptoms, 1):
#     plt.subplot(2, 3, i)
#     sns.countplot(x=symptom, hue='Diagnosis', data=df)
#     plt.title(f'{symptom} by Alzheimer\'s Diagnosis')
#     plt.tight_layout()
# plt.show()
#
# #Heatmap of Medical Conditions
# plt.figure(figsize=(10, 6))
# corr = df[['FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
#            'Depression', 'Hypertension', 'Diagnosis']].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title('Correlation of Medical Conditions')
# plt.show()

#correlation heatmap
# Select numerical columns for the correlation matrix
numerical_columns = df.select_dtypes(include=['float64', 'int64']).drop(columns=['PatientID']).columns

# Compute the correlation matrix
corr_matrix = df[numerical_columns].corr()

# Plot the heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, annot_kws={"size": 8})
plt.title('Correlation Heatmap of Numerical Variables', fontsize=16)
plt.show()

#
# Compute Pearson correlation coefficients
correlations = df.corr(numeric_only=True)['Diagnosis'][:-1].sort_values()

# Set the size of the figure
plt.figure(figsize=(20, 7))

# Create a bar plot of the Pearson correlation coefficients
ax = correlations.plot(kind='bar', width=0.7)

# Set the y-axis limits and labels
ax.set(ylim=[-1, 1], ylabel='Pearson Correlation', xlabel='Features',
       title='Pearson Correlation with Diagnosis')

# Rotate x-axis labels for better readability
ax.set_xticklabels(correlations.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

#data-processing
# Check for missing values in the dataset
missing_values = df.isnull().sum()

# Filter and display only columns with missing values
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("No missing values found in the dataset.")
else:
    print("Columns with missing values:")
    print(missing_values)

columns = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']

#normalize the columns
min_max_scaler = MinMaxScaler()
df[columns] = min_max_scaler.fit_transform(df[columns])

#standardize the columns
standard_scaler = StandardScaler()
df[columns] = standard_scaler.fit_transform(df[columns])


df_cleaned = df.drop(columns=['PatientID', 'DoctorInCharge'])

#split data into features and target
X = df_cleaned.drop(columns = ['Diagnosis'])
y = df_cleaned['Diagnosis']

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score
import numpy as np

# Define the data (assuming X and y are defined from the dataset)
X = df_cleaned.drop('Diagnosis', axis=1)  # Features
y = df_cleaned['Diagnosis']  # Target

# Instantiate the models with default parameters
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'XGBoost': XGBClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

# Define hyperparameter grids for each model
param_grids = {
    'Decision Tree': {'max_depth': [3, 5, 7, 12, None]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 12, None]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale', 'auto']},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7]},
    'CatBoost': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
}

# Set up Leave-One-Out Cross-Validation (LOOCV)
loo = LeaveOneOut()

# Define a custom F1 scorer to handle zero division gracefully
f1_scorer = make_scorer(f1_score, zero_division=1)

# Loop through models and perform LOOCV with hyperparameter tuning using GridSearchCV
for name, model in models.items():
    print(f'Processing {name}...')

    # Grid search with LOOCV
    grid_search = GridSearchCV(model, param_grids[name], cv=loo, scoring=f1_scorer, n_jobs=-1)

    # Fit the model using the entire dataset with LOOCV
    grid_search.fit(X, y)

    # Extract the best model and print the results
    best_model = grid_search.best_estimator_
    print(f'Best Parameters for {name}: {grid_search.best_params_}')

    # Since LOOCV uses all data for fitting, we directly print the cross-validation results
    mean_f1 = np.mean(grid_search.cv_results_['mean_test_score'])
    print(f'Average F1 Score for {name}: {mean_f1:.4f}\n')

