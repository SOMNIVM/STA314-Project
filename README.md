
# Background
- The diagnosis of Alzheimer is a typical binary classification problem
- We chose 4 models, including logistic regression, vector support machine, decision tree, and random forest for classification problems.
- The first three models were taught in class. 
- All of the models are taught in class

# Feature Selection
- We first did some early data analysis, including the scatter plot in the file EDA_Dinghui_Chen.py
- We observed the correlation scatterplot of all the numerical and categorical variables 
- The selected numerical variables based on the correlation are sleep quality, MMSE, Functional Assessment, and ADL.
- We select categorical variables based on previous researches.

# Coding
- 4 different models and their relevant are saved in their own directories named with the models' names.
- For each model, a file is created to fit and save the model in .joblib form. Then another file reads the test data and makes prediciton. 

# Result
- The final prediction is based on the 20-fold CV random forest model. Its in the "prediction" directory and named as "random_forest_submission_18_variables"


- The final accuracy rate of the 20-fold CV random forest prediction in Kaggle is 0.91987


# Analysis:
## For the 18 variables version:
- The best 10-folds cross validation accuracy scores for the 4 models are below:
- Logistic Regression: 0.8418
- Support Vector Machine: 0.8823
- Decision Tree: 0.9521
- Random Forest: 0.9555
- Logistic Regression (with regularization feature selection): 0.8398

# Other Attempts
Apart from the submitted prediction, we also tried other variations, including different predictors.


# Appendix
## For the 7 variables models:
we dropped: 

X = file.drop(columns=['PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                     'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diagnosis',
                     'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                     'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])

y = file['Diagnosis']

## For the 18 variables models:
we dropped:

X = file.drop(columns=['PatientID', 'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                     'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diagnosis',
                     'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                     'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])

y = file['Diagnosis']

## For the 32 variables models:
we dropped:

X = file.drop(columns=['PatientID', 'DoctorInCharge'])

y = file['Diagnosis']
