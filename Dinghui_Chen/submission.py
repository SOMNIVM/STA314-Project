import pandas as pd
import joblib

best_rf = joblib.load('best_random_forest_model.joblib')

# Load the test dataset
X_test = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/test.csv')

X_test = X_test.drop(columns=['Diagnosis', 'DoctorInCharge'], errors='ignore')

print("Test dataset size:", X_test.shape)

# Make predictions using the trained model
predictions = best_rf.predict(X_test)

submission = pd.DataFrame({
    'PatientID': X_test['PatientID'],
    'Prediction': predictions
})

# Save the submission file as CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created as 'submission.csv'")
