import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

data = {
    "Age": [22,25,47,52,46,56,23,34,42,29,31,50,48,26,39,41,58,33,28,45],
    "Income": [25000,32000,80000,95000,72000,110000,28000,45000,68000,39000,
               42000,88000,91000,31000,60000,70000,120000,50000,36000,85000],
    "Visits": [2,3,8,10,7,12,1,4,6,3,5,9,8,2,5,7,11,4,3,9],
    "High_Purchase": [0,0,1,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,0,1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df.head())

X = df[["Age", "Income", "Visits"]]
y = df["High_Purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = DecisionTreeClassifier(max_depth=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(6,4))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")
plt.show()

print("\n===== PROJECT SUMMARY =====")
print("Algorithm Used: Decision Tree Classifier")
print(f"Accuracy: {accuracy:.2%}")
print(f"AUC Score: {auc:.2f}")

print("\nGenerated Files:")
print("1. confusion_matrix.png")
print("2. roc_curve.png")
print("3. feature_importance.png")