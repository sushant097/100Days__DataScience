import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_decision_boundary

# Generate synthetic classification data
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42)

# Train AdaBoost Classifier
model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
model.fit(X, y)

# Visualization
plot_decision_boundary(model, X, y)
plt.title("Boosting with AdaBoost")
plt.show()
