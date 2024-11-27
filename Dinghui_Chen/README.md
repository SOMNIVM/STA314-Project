
# Background
- The diagnosis of Alzheimer is a typical binary classification problem
- We chose 4 models, including logistic regression, vector support machine, decision tree, and random forest for classification problems.
- The first three models were taught in class. 
- The last one is the extension of decision tree. Therefore we also include it after self-studying.

# Feature Selection
- We first did some early data analysis, including the scatter plot in the file EDA_Dinghui_Chen.py
- We observed the correlation scatterplot of all the numerical and categorical variables 
- The selected numerical variables based on the correlation are sleep quality, MMSE, Functional Assessment, and ADL.
- We select categorical variables based on previous researches.


# Analysis:
## For the 18 variables version:
- The best 10-folds cross validation accuracy scores for the 4 models are below:
- Logistic Regression: 0.8418
- Support Vector Machine: 0.8823
- Decision Tree: 0.9521
- Random Forest: 0.9555

## For the models with regularization feature selection:
Logistic Regression: 0.8398

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
