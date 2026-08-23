import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df =pd.read_csv("Level_1/Level_2/dataset2.csv")
print("First 5 rows of the dataset:")
print (df.head())
print("\nDataset Information:")
print("df.info()")
print("\nMissing  Values:")
print(df.isnull().sum())
print("\nSummary Statistics:")
print(df.describe())
print("\nColumn Names:")
print(df.columns)
df["Churn"] = df["Churn"].astype(int)
df = pd.get_dummies(df,drop_first=True)
X = df.drop("Churn", axis=1)
y = df["Churn"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2,random_state=42)
print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")
y_pred = model.predict(X_test)
print("Predictions:")
print(y_pred[:10])
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print("R2_score:", r2)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.savefig("regression_results.png")
plt.show()


