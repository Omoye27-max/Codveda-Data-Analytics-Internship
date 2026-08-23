import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Level_2/dataset2.csv")
print(df.head())
print(df.info())
print("\nDataset Information:")
print("\nMissing Values:")
print(df.isnull().sum())
df = df.drop("Churn", axis=1)
df = pd.get_dummies(df,drop_first=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
print("Data scaled successfully")
wcss = [] 
for i in range(1, 11):
    kMeans = KMeans(n_clusters=i,random_state=42, n_init=10)
    kMeans.fit(scaled_data) 
    wcss.append(kMeans.inertia_) 
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11),wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("elbow_plot.png")
kMeans = KMeans(n_clusters=3,random_state=42, n_init=10)
df["Cluster"] = kMeans.fit_predict(scaled_data)
print(df.head())
print(df.groupby("Cluster").mean())
df.to_csv("clustered_data.csv", index=False)
print("Clustered data saved successfully!")