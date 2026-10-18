
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("data/cardio_train (1) (2).csv", sep=";")

# Clean invalid blood pressure records
clean_data = data[
    (data["ap_hi"] > 0) &
    (data["ap_lo"] > 0) &
    (data["ap_hi"] <= 250) &
    (data["ap_lo"] <= 200) &
    (data["ap_hi"] >= data["ap_lo"])
].copy()

# Prepare features and target
X = clean_data.drop(["id", "cardio"], axis=1)
y = clean_data["cardio"]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X, y)

# Save model
joblib.dump(model, "cardio_model.pkl")

print("Model saved successfully!")
