import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Read the data
data_set = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# Now 'data_set' contains the dataset
data_set.head()

# Data Cleaning

# Drop column = 'id'
data_set.drop(columns='id', inplace=True)
# Data Cleaning (Continued)

# Check for missing values
print(data_set.isnull().sum())

# Fill missing values in 'bmi' column with mean
data_set['bmi'].fillna(data_set['bmi'].mean(), inplace=True)

# Convert categorical variables to numerical using Label Encoding
le = LabelEncoder()
data_set['gender'] = le.fit_transform(data_set['gender'])
data_set['ever_married'] = le.fit_transform(data_set['ever_married'])
data_set['work_type'] = le.fit_transform(data_set['work_type'])
data_set['Residence_type'] = le.fit_transform(data_set['Residence_type'])
data_set['smoking_status'] = le.fit_transform(data_set['smoking_status'])

# Data Preprocessing

# Split data into features and target variable
X = data_set.drop(columns='stroke')
y = data_set['stroke']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handling Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Training the models

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_resampled, y_train_resampled)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_resampled, y_train_resampled)

# Evaluating the models

# Logistic Regression
log_reg_pred = log_reg.predict(X_test)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, log_reg_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_reg_pred))
print("Classification Report:\n", classification_report(y_test, log_reg_pred))

# Random Forest
rf_pred = rf.predict(X_test)
print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Support Vector Machine (SVM)
svm_pred = svm.predict(X_test)
print("\nSupport Vector Machine:")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))

# Hybrid Prediction

# Combine predictions from all models
from scipy.stats import mode
combined_pred = np.array([log_reg_pred, rf_pred, svm_pred])
final_pred, _ = mode(combined_pred)

# Calculate accuracy of the hybrid model
final_pred = final_pred.reshape(-1)
print("\nHybrid Model Accuracy:", accuracy_score(y_test, final_pred))

