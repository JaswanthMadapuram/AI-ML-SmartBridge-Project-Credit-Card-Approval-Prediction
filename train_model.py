import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#APPLICATION DATASET
application_df = pd.read_csv("dataset/application_record.csv")
credit_df = pd.read_csv("dataset/credit_record.csv")

#CREDIT DATASET
print("=" * 50)
print("APPLICATION DATASET")
print("=" * 50)

print(application_df.head())

print("\nShape:", application_df.shape)

print("\nColumns:")
print(application_df.columns)

print("\n")
print("=" * 50)
print("CREDIT DATASET")
print("=" * 50)

print(credit_df.head())

print("\nShape:", credit_df.shape)

print("\nColumns:")
print(credit_df.columns)

print("\nMissing Values in Application Dataset")
print(application_df.isnull().sum())
print("\nMissing Values in Credit Dataset")
print(credit_df.isnull().sum())

print("\n" + "=" * 50)
print("STATUS VALUE COUNTS")
print("=" * 50)
print(credit_df["STATUS"].value_counts())

# Convert STATUS into Good (0) and Bad (1)
credit_df["TARGET"] = credit_df["STATUS"].apply(
    lambda x: 1 if x in ["1", "2", "3", "4", "5"] else 0
)
print("\nTARGET VALUE COUNTS")
print(credit_df["TARGET"].value_counts())

# Keep one TARGET value per customer
credit_target = credit_df.groupby("ID")["TARGET"].max().reset_index()

print("\nCredit Target Shape:")
print(credit_target.shape)

# Merge with application dataset
final_df = application_df.merge(credit_target, on="ID", how="inner")
print("\nMerged Dataset Shape:")
print(final_df.shape)
print("\nFirst 5 Rows:")
print(final_df.head())


print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)
# Fill missing values
final_df["OCCUPATION_TYPE"] = final_df["OCCUPATION_TYPE"].fillna("Unknown")

""" Encode categorical columns
le = LabelEncoder()
for col in final_df.columns:
    if final_df[col].dtype == "object":
        final_df[col] = le.fit_transform(final_df[col])"""

# Remove ID column
final_df = final_df.drop("ID", axis=1)
print("\nDataset after preprocessing:")
print(final_df.head())
print("\nDataset Shape:")
print(final_df.shape)

print("\n" + "=" * 50)
print("FEATURES AND TARGET")
print("=" * 50)
selected_features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "AMT_INCOME_TOTAL",
    "CNT_CHILDREN",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS"
]

X = final_df[selected_features]
y = final_df["TARGET"]
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

#Filling Missing Values
X = X.fillna("Unknown")

# Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)
print("\nAfter Encoding:")
print(X.head())
print("\nEncoded Feature Shape:")
print(X.shape)

print("\n" + "=" * 50)
print("TRAIN TEST SPLIT")
print("=" * 50)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\n" + "="*50)
print("SMOTE APPLIED")
print("="*50)

print("Original Training Shape :", X_train.shape)
print("Balanced Training Shape :", X_train_smote.shape)

print("\nBalanced Target Counts:")
print(y_train_smote.value_counts())
print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)
print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)

print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train_smote, y_train_smote)
print("All Models Trained Successfully!")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

print("\nPrediction Counts")
print(pd.Series(rf_pred).value_counts())
print("\nActual Counts")
print(y_test.value_counts())

print("\n" + "="*50)
print("MODEL ACCURACY")
print("="*50)

print("Logistic Regression Accuracy :", accuracy_score(y_test, lr_pred))
print("Decision Tree Accuracy       :", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy       :", accuracy_score(y_test, rf_pred))

print("\n" + "="*50)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("="*50)

print(classification_report(y_test, rf_pred))
print("\nPrediction Probabilities")
print(rf.predict_proba(X_test[:10]))
print("\nModel Classes")
print(rf.classes_)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nSample Predictions")

sample = X_test.iloc[:20]

sample_pred = rf.predict(sample)

sample_prob = rf.predict_proba(sample)

for i in range(len(sample)):
    print(
        f"Customer {i+1} -> Prediction={sample_pred[i]}, "
        f"Approved={sample_prob[i][0]*100:.2f}%, "
        f"Rejected={sample_prob[i][1]*100:.2f}%"
    )

# ==========================================
# SAVE BEST MODEL
# ==========================================

import joblib
joblib.dump(rf, "credit_card_approval_model.pkl")
print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)
print("Saved as: credit_card_approval_model.pkl")

import joblib

# Save feature names
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
print("Feature columns saved successfully!")
