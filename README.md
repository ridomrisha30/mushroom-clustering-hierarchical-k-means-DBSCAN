# Mushroom Clustering: Hierarchical, K-Means & DBSCAN

## Project Overview
This project applies Hierarchical Clustering, K-Means, and DBSCAN to the Mushroom Dataset to uncover natural groupings based on mushroom characteristics. The dataset contains categorical features, making it an interesting case for clustering methods that typically handle numerical data.

## Dataset
* The dataset is available in this repository: [`mushrooms.csv`](./mushrooms.csv)
* Size: 8124 rows, 22 categorical features (e.g., cap shape, gill color, odor)
### Preprocessing:
* Categorical encoding using Label Encoding
* Feature scaling using StandardScaler
* PCA (2 components) for visualization

## Clustering Methods Used
1️. Hierarchical Clustering (Agglomerative, Ward's Method)
* Forms a tree-like structure (dendrogram) to determine cluster relationships
* Allows flexible, non-spherical clusters
* Doesn't require predefined k

2️. K-Means Clustering
* Groups data based on centroid distances
* Requires predefined number of clusters (k)
* Assumes spherical clusters (which may not always be suitable)

3️. DBSCAN (Density-Based Spatial Clustering)
* Groups high-density regions and ignores noise
* Automatically detects clusters without predefined k
* Struggles with categorical & high-dimensional data


## Visualizations  

![Hierarchical Clustering - PCA Plot](hierarchical_clusters.png)

### **1. Hierarchical Clustering (Dendrogram)**
![Hierarchical Clustering - Dendrogram](hierarchical_dendrogram.png)

### **2. Hierarchical Clustering - PCA Scatter Plot**
![Hierarchical Clustering - PCA Plot](hierarchical_clusters.png)

### **3. K-Means Clustering - PCA Scatter Plot**
![K-Means Clustering - PCA Plot](kmeans_clusters.png)

### **4. DBSCAN Clustering - PCA Scatter Plot**
![DBSCAN Clustering - PCA Plot](dbscan_clusters.png)

## Performance Evaluation
* Hierarchical & K-Means performed well, but DBSCAN struggled due to categorical data and lack of density variations.
* The Adjusted Rand Index (ARI) between Hierarchical and K-Means is {ari_hierarchical_kmeans:.4f}, showing that both methods produced similar cluster assignments.

## Outputs
* Clustered dataset saved: mushroom_clusters.csv
* Visualizations:
  * Dendrogram for Hierarchical Clustering
  * PCA-based scatter plot with cluster assignments

