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
# from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)

train_data = pd.read_csv('C:/Users/17764/PycharmProjects/STA314-Project/train.csv')

# Continue with the correlation matrix and scatterplot
train_data_cleaned = train_data.drop(columns=['PatientID', 'DoctorInCharge', 'Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'Depression', 'HeadInjury', 'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness', 'Hypertension', 'CardiovascularDisease', 'Diabetes'], errors='ignore')

# Compute the correlation matrix
correlation_matrix = train_data_cleaned.corr()

# Plot the scatterplot of the correlation matrix using a heatmap
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True,
            annot_kws={"size": 6, "fontweight": "regular", "color": "black"},)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title("Correlation Matrix of Training Data with Values", fontsize=12)
plt.show()

# Plot the personal correlation coefficients
# Compute Pearson correlation coefficients
correlations = train_data_cleaned.corr(numeric_only=True)['Diagnosis'][:-1].sort_values()

# Set the size of the figure
plt.figure(figsize=(20, 15))

# Create a bar plot of the Pearson correlation coefficients
ax = correlations.plot(kind='bar', width=0.7)

# Set the y-axis limits and labels
ax.set(ylim=[-1, 1], ylabel='Pearson Correlation', xlabel='Features',
       title='Pearson Correlation with Diagnosis')

ax.set_xticklabels(correlations.index, rotation=45, ha='right')
plt.tight_layout()
plt.show()
