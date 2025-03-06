import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("creditcard.csv")

# Extract relevant features
df["Transaction_Hour"] = (df["Time"] // 3600) % 24  # Convert seconds to hours
selected_features = ["Time", "Amount", "Transaction_Hour"]
X = df[selected_features]
y = df["Class"]  # Fraud or Not Fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Save the model
with open("fraud_detection_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
