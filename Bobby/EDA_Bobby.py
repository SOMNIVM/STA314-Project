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
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)

df = pd.read_csv('/Users/bobbybao/PycharmProjects/STA314-Project/train.csv')
df.info()

df.describe().T

# Display the first few rows of the dataframe
print(df.head())

# Summary of the dataset's structure
print(df.info())

# Statistical summary of numeric columns
print(df.describe().T)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Visualize the distribution of each feature using histograms
df.hist(figsize=(20, 15))
plt.show()

# Box plots to visualize outliers
df.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False, figsize=(20,15))
plt.show()

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df.sample(500))  # using sample for efficiency if the dataset is large
plt.show()

# Scatter plots of important feature pairs (you can specify pairs you're interested in)
# Example: plot Age vs. Fare in the Titanic dataset
# sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
# plt.show()

