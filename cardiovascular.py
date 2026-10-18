import pandas as pd

data = pd.read_csv("data/cardio_train (1) (2).csv", sep=";")

print("COLUMN NAMES:")
print(data.columns.tolist())

print(data.shape)
print(data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())
print(data["cardio"].value_counts())


X = data.drop(["id", "cardio"], axis=1)
y = data["cardio"]

print("Input data shape:", X.shape)
print("Target data shape:", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training inputs:", X_train.shape)
print("Testing inputs:", X_test.shape)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model training completed!")


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)
print("Model accuracy percentage:", accuracy * 100, "%")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("New accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Random Forest accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("Feature importance:")
print(importance)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print(data[["ap_hi", "ap_lo"]].describe())

print("Negative or zero systolic readings:", (data["ap_hi"] <= 0).sum())
print("Negative or zero diastolic readings:", (data["ap_lo"] <= 0).sum())
print("Systolic readings above 250:", (data["ap_hi"] > 250).sum())
print("Diastolic readings above 200:", (data["ap_lo"] > 200).sum())
print("Systolic below diastolic:", (data["ap_hi"] < data["ap_lo"]).sum())

suspicious = data[
    (data["ap_hi"] <= 0) |
    (data["ap_lo"] <= 0) |
    (data["ap_hi"] > 250) |
    (data["ap_lo"] > 200) |
    (data["ap_hi"] < data["ap_lo"])
]

print("Total suspicious rows:", len(suspicious))

suspicious = data[
    (data["ap_hi"] <= 0) |
    (data["ap_lo"] <= 0) |
    (data["ap_hi"] > 250) |
    (data["ap_lo"] > 200) |
    (data["ap_hi"] < data["ap_lo"])
]

print("Total suspicious rows:", len(suspicious))

clean_data = data[
    (data["ap_hi"] > 0) &
    (data["ap_lo"] > 0) &
    (data["ap_hi"] <= 250) &
    (data["ap_lo"] <= 200) &
    (data["ap_hi"] >= data["ap_lo"])
].copy()

print("Original rows:", len(data))
print("Rows after cleaning:", len(clean_data))
print("Rows removed:", len(data) - len(clean_data))

X = clean_data.drop(["id", "cardio"], axis=1)
y = clean_data["cardio"]

print("Input features shape:", X.shape)
print("Target shape:", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training inputs:", X_train.shape)
print("Testing inputs:", X_test.shape)
print("Training answers:", y_train.shape)
print("Testing answers:", y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy:", accuracy)
print("Model accuracy percentage:", accuracy * 100, "%")



from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_pred)

print("Decision Tree accuracy:", tree_accuracy)
print("Decision Tree accuracy percentage:", tree_accuracy * 100, "%")



from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)

forest_accuracy = accuracy_score(y_test, forest_pred)

print("Random Forest accuracy:", forest_accuracy)
print("Random Forest accuracy percentage:", forest_accuracy * 100, "%")



from sklearn.metrics import classification_report

print("Random Forest Classification Report:")
print(classification_report(y_test, forest_pred))



import pandas as pd

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": forest_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test,
    forest_pred,
    display_labels=["No disease recorded", "Disease recorded"]
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()



sample = X_test.iloc[[0]]

prediction = forest_model.predict(sample)

if prediction[0] == 1:
    print("Model prediction: Cardiovascular disease recorded")
else:
    print("Model prediction: No cardiovascular disease recorded")

print("Actual dataset label:", y_test.iloc[0])


# Correlation Matrix
corr = clean_data.drop(columns=["id"]).corr()

plt.figure(figsize=(12, 9))

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.colorbar(label="Correlation")
plt.title("Correlation Matrix of Cardiovascular Dataset")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))

clean_data["cardio"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Cardiovascular Disease Class")
plt.ylabel("Number of Records")
plt.title("Distribution of Cardiovascular Disease Classes")
plt.xticks([0, 1], ["No disease recorded (0)", "Disease recorded (1)"], rotation=0)

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))

plt.hist(clean_data["age"] / 365.25, bins=30, edgecolor="black")

plt.xlabel("Age (years)")
plt.ylabel("Number of Records")
plt.title("Age Distribution of the Dataset")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.hist(
    clean_data["ap_hi"],
    bins=40,
    alpha=0.6,
    label="Systolic (ap_hi)"
)

plt.hist(
    clean_data["ap_lo"],
    bins=40,
    alpha=0.6,
    label="Diastolic (ap_lo)"
)

plt.xlabel("Blood Pressure Value")
plt.ylabel("Number of Records")
plt.title("Blood Pressure Distribution")
plt.legend()

plt.tight_layout()
plt.show()


cholesterol_counts = clean_data.groupby(
    ["cholesterol", "cardio"]
).size().unstack(fill_value=0)

cholesterol_counts.plot(kind="bar", figsize=(8, 5))

plt.xlabel("Cholesterol Level")
plt.ylabel("Number of Records")
plt.title("Cholesterol Levels by Cardiovascular Disease Class")
plt.xticks(rotation=0)
plt.legend(["No disease recorded (0)", "Disease recorded (1)"])

plt.tight_layout()
plt.show()


age_by_cardio = clean_data.groupby("cardio")["age"].mean() / 365.25

plt.figure(figsize=(6, 4))

age_by_cardio.plot(kind="bar")

plt.xlabel("Cardiovascular Disease Class")
plt.ylabel("Average Age (years)")
plt.title("Average Age by Cardiovascular Disease Class")
plt.xticks([0, 1], ["No disease recorded (0)", "Disease recorded (1)"], rotation=0)

plt.tight_layout()
plt.show()


# SVM Model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create the SVM model
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf")
)

# Train the model
print("Training SVM model... Please wait.")

svm_model.fit(X_train, y_train)

# Make predictions
svm_pred = svm_model.predict(X_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)

# Display results
print("\n----- SVM RESULTS -----")
print("SVM accuracy:", round(svm_accuracy, 4))
print("SVM accuracy percentage:", round(svm_accuracy * 100, 2), "%")

# KNN Model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Training KNN model... Please wait.")

knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\n----- KNN RESULTS -----")
print("KNN accuracy:", round(knn_accuracy, 4))
print("KNN accuracy percentage:", round(knn_accuracy * 100, 2), "%")

# Compare All Five Machine Learning Models

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ],
    "Accuracy (%)": [
        accuracy * 100,
        tree_accuracy * 100,
        forest_accuracy * 100,
        svm_accuracy * 100,
        knn_accuracy * 100
    ]
})

results = results.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\n===== MODEL ACCURACY COMPARISON =====")
print(results.to_string(index=False))

# Model Accuracy Comparison Graph

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.bar(
    results["Model"],
    results["Accuracy (%)"]
)

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.title("Comparison of Machine Learning Model Accuracy")

plt.ylim(0, 100)
plt.xticks(rotation=20)

for i, value in enumerate(results["Accuracy (%)"]):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# Final Model Selection

best_model_name = results.iloc[0]["Model"]
best_accuracy = results.iloc[0]["Accuracy (%)"]

print("\n===== FINAL MODEL SELECTION =====")
print("Best performing model:", best_model_name)
print("Accuracy:", round(best_accuracy, 2), "%")








