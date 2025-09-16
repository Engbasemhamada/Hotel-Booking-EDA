
# 1. Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
data = pd.read_csv("first inten project.csv")
data.columns = data.columns.str.strip() # Good practice!

# 3. Initial Data Cleaning (Handling Missing Values)
# Fill missing numeric values with median
numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# 4. Define Features (X) and Target (y)
X = data.drop("booking status", axis=1)
y = data["booking status"]

# Encode the target variable (y) because it's the output
le = LabelEncoder()
y = le.fit_transform(y)

# 5. Train-Test Split (MOST IMPORTANT: DO THIS FIRST!)
# This prevents data leakage from preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Preprocessing Pipeline
# Identify column types from the TRAINING data
numeric_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(include='object').columns

# Create preprocessing steps for each type
# For numeric data: scale them
numeric_transformer = StandardScaler()

# For categorical data: one-hot encode them
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # 'ignore' helps with new categories in test set

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns if any (not recommended usually)
)


# 7. Modeling
# We will create a full pipeline for each model
# This pipeline will first preprocess the data, then train the model

# --- Logistic Regression Model ---
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# --- Random Forest Model ---
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 8. Train the Models
print("Training Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)


# 9. Evaluation
# Make predictions
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)

# Print reports
print("\n Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


import pickle

with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(lr_pipeline, file)

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(lr_pipeline, file)





