from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
import matplotlib.pyplot as plt

# Example: Classification Metrics
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1]
y_pred = [1, 0, 1, 1, 0, 0, 0, 1, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-Score:", f1_score(y_true, y_pred))

# Example: Regression Metrics
y_true_reg = [3.0, -0.5, 2.0, 7.0]
y_pred_reg = [2.5, 0.0, 2.1, 7.8]

print("MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
print("R2 Score:", r2_score(y_true_reg, y_pred_reg))


# Generate synthetic data for clustering
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Compute clustering metrics
silhouette = silhouette_score(X, y_kmeans)
davies_bouldin = davies_bouldin_score(X, y_kmeans)

# Visualization of clustering results
plt.figure(figsize=(12, 6))

# Original clusters (Ground truth)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.8)
plt.title("Ground Truth Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# KMeans Clustering results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=30, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=100, marker='x', label="Centroids")
plt.title(f"KMeans Clustering\nSilhouette Score: {silhouette:.2f} | Davies-Bouldin Index: {davies_bouldin:.2f}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

plt.tight_layout()
plt.show()

# Print metric values
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")