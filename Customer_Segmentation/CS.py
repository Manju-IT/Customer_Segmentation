import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset from Task 1
df = pd.read_excel("Customer Segmentation and Product Recommendation System for GTBank.xlsx")

# Select clustering variables (justified below)
cluster_vars = [
    'Age', 
    'Average Monthly Balance', 
    'Transaction Frequency (per month)', 
    'Service Feedback Score'
]

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[cluster_vars])

# Elbow Method
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Distances (SSE)")
plt.title("Elbow Method for Optimal k")
plt.grid()
plt.savefig('elbow_plot.png', dpi=300)
plt.show()


### K-Means Clustering

# Fit K-Means with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Cluster Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', 
    hue='Cluster', 
    palette='viridis', 
    data=df,
    s=100
)
plt.title("Customer Clusters (PCA Visualization)")
plt.savefig('cluster_pca.png', dpi=300)
plt.show()

## Segment Analysis

cluster_summary = df.groupby('Cluster')[cluster_vars].mean().reset_index()
print(cluster_summary)

# Boxplots for cluster comparison
plt.figure(figsize=(15, 10))
for i, var in enumerate(cluster_vars, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Cluster', y=var, data=df, palette='viridis')
    plt.title(f'Distribution of {var} by Cluster')
plt.tight_layout()
plt.savefig('cluster_distributions.png', dpi=300)
plt.show()


