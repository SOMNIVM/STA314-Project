import pandas as pd
import joblib

# Load the trained Random Forest model
best_rf = joblib.load('../random_forest/best_random_forest_model_Gini_importance.joblib')

# Load the test dataset
X_test = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/test.csv')

# Retain only the selected features used in the model
top_10_features = [
    'FunctionalAssessment', 'ADL', 'MMSE', 'MemoryComplaints', 'BehavioralProblems',
    'SleepQuality', 'DietQuality', 'CholesterolTriglycerides', 'CholesterolHDL', 'CholesterolTotal'
]

X_test = X_test[top_10_features]

# Make predictions using the trained model
predictions = best_rf.predict(X_test)

submission = pd.DataFrame({
    'PatientID': pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/test.csv')['PatientID'],
    'Prediction': predictions
})

# Save the submission file as CSV
submission.to_csv('random_forest_submission.csv', index=False)
print("Submission file created as 'random_forest_submission.csv'")
