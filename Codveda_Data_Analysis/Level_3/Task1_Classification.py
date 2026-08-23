import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("Level_3/1) iris.csv")
print("First 5 Rows")
print(df.head())
print("\nDataset Information")
print(df.info())
print("\nSummary Statistics")
print(df.describe())
print("\nMissing Values")
print(df.isnull().sum())
print("\nColumn Names")
print(df.columns)
x = df.drop("species", axis=1)
y = df["species"]
print("Features shape:", x.shape)
print("Target shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("Training data:",X_train.shape)
print("Testing data:", X_test.shape)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("Model trained sucessfully")
y_pred = model.predict(X_test)
print("Predictions:")
print(y_pred)
accuracy = accuracy_score(y_test,y_pred)
print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(6,4))
df["species"].value_counts().plot(kind="bar")
plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.savefig("iris_species_distribution.png")
plt.show()























