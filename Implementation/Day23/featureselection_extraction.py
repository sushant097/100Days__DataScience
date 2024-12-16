import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Load dataset
data = load_iris()
X = data.data
y = data.target

# Feature Selection: Select top 2 features based on ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# Feature Extraction: Reduce to 2 components using PCA
pca = PCA(n_components=2)
X_extracted = pca.fit_transform(X)

# Convert to DataFrames for better visualization
selected_df = pd.DataFrame(X_selected, columns=["Feature 1", "Feature 2"])
extracted_df = pd.DataFrame(X_extracted, columns=["Component 1", "Component 2"])

# Plot comparision
plt.figure(figsize=(12, 6))

# Feature Selection Visualization
plt.subplot(1, 2, 1)
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=y, cmap='viridis', s=30, alpha=0.8)
plt.title("Feature Selection (ANOVA F-test)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Feature Extraction Visualizaiton
plt.subplot(1, 2, 2)
plt.scatter(X_extracted[:, 0], X_extracted[:, 1], c=y, cmap='viridis', s=30, alpha=0.8)
plt.title("Feature Extraction (PCA)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.tight_layout()
plt.show()

# Print DataFrames
print("Selected Features (Feature Selection): \n", selected_df.head())
print("Extracted Features (Feature Extraction): \n", extracted_df.head())

