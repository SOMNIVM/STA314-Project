import pandas as pd
import joblib

#best_rf = joblib.load('../random_forest/best_random_forest_model.joblib')

best_rf = joblib.load('../random_forest/best_random_forest_model.joblib')

# Load the test dataset
X_test = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/test.csv')

patient_ids = X_test['PatientID']
X_test = X_test.drop(columns=['PatientID', 'Age', 'Smoking', 'Depression', 'HeadInjury', 'Confusion', 'Disorientation', 'DifficultyCompletingTasks', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                     'DietQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
                     'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
                     'CholesterolHDL', 'CholesterolTriglycerides', 'DoctorInCharge'])

print("Test dataset size:", X_test.shape)

# Make predictions using the trained model
predictions = best_rf.predict(X_test)

submission = pd.DataFrame({
    'PatientID': patient_ids,
    'Prediction': predictions
})

# Save the submission file as CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created as 'random_forest_submission.csv'")
