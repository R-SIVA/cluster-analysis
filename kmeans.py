# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset

df = pd.read_csv("E:/Educational content/Sem5/Machine Learning/cluster-analysis/Mall_Customers.csv")


# Step 2: Data exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Plot histograms for Age, Annual Income, and Spending Score
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(bins=20, figsize=(10, 7))
plt.show()

# Step 3: Data Preprocessing
# Dropping unnecessary columns like CustomerID and Gender
df_processed = df.drop(columns=['CustomerID', 'Gender'])

# Normalize the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_processed)

# Convert scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=df_processed.columns)

# Step 4: Choosing the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal number of clusters')
plt.show()

# Step 5: Train the KMeans model with 5 clusters (based on the elbow graph)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

print("\nCluster Assignments:")
print(df.head())

# Step 6: Visualizing Clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], 
                hue=df['Cluster'], palette='viridis', s=100)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.show()

# Step 7: Cluster Analysis
cluster_centers = kmeans.cluster_centers_
cluster_df = pd.DataFrame(scaler.inverse_transform(cluster_centers), columns=df_processed.columns)

print("\nCluster Centers:")
print(cluster_df)

# Checking the mean values for each cluster
cluster_summary = df.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)

# Step 8: Model Evaluation Using Silhouette Score
silhouette_avg = silhouette_score(scaled_df, df['Cluster'])
print(f'\nSilhouette Score: {silhouette_avg:.2f}')
