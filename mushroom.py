import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load dataset
csv_path = r"C:\Users\grigt\Downloads\archive (3)\mushrooms.csv"
df = pd.read_csv(csv_path)

# Drop target column ('class') to make it unsupervised
df.drop(columns=['class'], inplace=True)

# Encode categorical features
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Perform Hierarchical Clustering
linked = linkage(data_pca, method='ward')

# Determine optimal number of clusters (e.g., 5 clusters)
n_clusters = 5
clusters_hierarchical = fcluster(linked, n_clusters, criterion='maxclust')

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(data_pca)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_clusters = dbscan.fit_predict(data_pca)

# Evaluate clustering with silhouette score
silhouette_hierarchical = silhouette_score(data_pca, clusters_hierarchical)
silhouette_kmeans = silhouette_score(data_pca, kmeans_clusters)
silhouette_dbscan = silhouette_score(data_pca, dbscan_clusters) if len(set(dbscan_clusters)) > 1 else "N/A"

print(f"Silhouette Score - Hierarchical: {silhouette_hierarchical:.4f}")
print(f"Silhouette Score - K-Means: {silhouette_kmeans:.4f}")
print(f"Silhouette Score - DBSCAN: {silhouette_dbscan}")

# Compare Hierarchical vs K-Means using Adjusted Rand Index (ARI)
ari_hierarchical_kmeans = adjusted_rand_score(clusters_hierarchical, kmeans_clusters)
print(f"Adjusted Rand Index (Hierarchical vs K-Means): {ari_hierarchical_kmeans:.4f}")

# Add cluster labels to original dataset
df['Hierarchical_Cluster'] = clusters_hierarchical
df['KMeans_Cluster'] = kmeans_clusters
df['DBSCAN_Cluster'] = dbscan_clusters

# Save clustered dataset
df.to_csv(r"C:\Users\grigt\Downloads\archive (3)\mushrooms.csv", index=False)
print("Clustered dataset saved as mushroom_clusters.csv")

# Visualization
#hierarchical plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=clusters_hierarchical, palette='viridis')
plt.title("Hierarchical Clustering of Mushroom Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.savefig("hierarchical_clusters.png") 
plt.show()

#kmeans plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=kmeans_clusters, palette='coolwarm')
plt.title("K-Means Clustering of Mushroom Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("kmeans_clusters.png")  # Save for GitHub
plt.show()

#dbscan plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=dbscan_clusters, palette='Set1')
plt.title("DBSCAN Clustering of Mushroom Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.savefig("dbscan_clusters.png")  # Save for GitHub
plt.show()

