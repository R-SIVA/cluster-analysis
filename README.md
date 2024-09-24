# Cluster Analysis


### **1. Importing Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
```

- **pandas**: Used for handling the dataset in a structured format (DataFrames).
- **numpy**: Useful for numerical operations and array manipulation.
- **matplotlib** and **seaborn**: Used for visualizing the data with plots and graphs.
- **sklearn.cluster.KMeans**: The algorithm used for clustering.
- **StandardScaler**: Used to normalize the features so that they all have a similar scale.
- **silhouette_score**: Used to evaluate how well the clusters are separated from one another.

---

### **2. Loading the Dataset**

```python
df = pd.read_csv("E:/Educational content/Sem5/Machine Learning/cluster-analysis/Mall_Customers.csv")
```

- We load the **Mall Customers Dataset**. This dataset contains information about customers, such as their Age, Gender, Annual Income, and Spending Score.

---

### **3. Data Exploration**

```python
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nStatistical Summary:")
print(df.describe())
```

- **df.head()**: Shows the first 5 rows of the dataset so we can get an idea of its structure.
- **df.info()**: Provides details about the dataset like column names, data types, and whether or not there are any missing values.
- **df.describe()**: Gives statistical insights into numerical columns (like mean, standard deviation, etc.).

---

### **4. Visualizing Features**

```python
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(bins=20, figsize=(10, 7))
plt.show()
```

- We visualize the distribution of the key features (`Age`, `Annual Income`, and `Spending Score`) using histograms. This helps in understanding the spread of the data (e.g., are certain groups over-represented?).

---

### **5. Data Preprocessing**

```python
df_processed = df.drop(columns=['CustomerID', 'Gender'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_processed)
```

- **Dropping unnecessary columns**: We drop `CustomerID` because it's just an identifier and doesn’t contribute to clustering. We also drop `Gender` as it is categorical, and we're focusing on numerical features here.
- **StandardScaler**: We normalize the data so that each feature has a mean of 0 and a standard deviation of 1. This is important for K-Means because it’s sensitive to feature scaling (e.g., features with larger values could dominate the clustering if left unscaled).

---

### **6. Using the Elbow Method to Choose the Optimal Number of Clusters**

```python
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)
```

- **K-Means and inertia**: We run the K-Means algorithm for different values of `k` (the number of clusters) and store the **inertia** (which measures how well the data points are clustered around their centroids).
- **Elbow Method**: We plot inertia against the number of clusters. The "elbow" in the plot indicates the optimal number of clusters, where adding more clusters doesn’t significantly reduce inertia.

```python
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal number of clusters')
plt.show()
```
- The elbow plot helps us visually determine the optimal number of clusters by showing where the curve bends or “elbows.”

---

### **7. Training the KMeans Model**

```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)
```

- After determining that **5 clusters** is optimal from the Elbow Method, we fit the K-Means algorithm using `n_clusters=5`. 
- **fit_predict()**: This not only fits the model but also assigns each data point to a cluster, which we store in a new column `Cluster` in the original DataFrame.

---

### **8. Visualizing the Clusters**

```python
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], 
                hue=df['Cluster'], palette='viridis', s=100)
plt.title('Customer Segments based on Annual Income and Spending Score')
plt.show()
```

- **Scatterplot**: We visualize how the customers are clustered based on their `Annual Income` and `Spending Score`. 
- **hue='Cluster'**: This colors the points according to which cluster they belong to, helping us see the segmentation.

---

### **9. Analyzing the Cluster Centers**

```python
cluster_centers = kmeans.cluster_centers_
cluster_df = pd.DataFrame(scaler.inverse_transform(cluster_centers), columns=df_processed.columns)

print("\nCluster Centers:")
print(cluster_df)
```

- **Cluster Centers**: After clustering, K-Means provides the centroids of each cluster. We transform these back into the original scale (undoing the normalization) so we can interpret them more easily.
- **cluster_df**: This shows the representative "center" of each cluster for features like `Age`, `Annual Income`, and `Spending Score`.

---

### **10. Cluster Summary**

```python
cluster_summary = df.groupby('Cluster').mean()
print("\nCluster Summary:")
print(cluster_summary)
```

- **groupby('Cluster')**: We group the data by cluster to examine the average characteristics (mean) of each cluster. This helps us interpret each cluster’s demographic and spending behavior.

---

### **11. Model Evaluation with Silhouette Score**

```python
silhouette_avg = silhouette_score(scaled_df, df['Cluster'])
print(f'\nSilhouette Score: {silhouette_avg:.2f}')
```

- **Silhouette Score**: This is a metric to evaluate the quality of clustering. It measures how similar each point is to its own cluster (cohesion) compared to other clusters (separation). The score ranges from -1 to 1:
  - **+1**: Perfect separation between clusters.
  - **0**: Overlapping clusters.
  - **-1**: Incorrect clustering.
  
  A silhouette score closer to 1 indicates good clustering quality.
