import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Predict and visualize
X_test = np.linspace(min(X), max(X), 500).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_test, y_pred, color="red", label="Bagging Prediction")
plt.legend()
plt.title("Bagging with Random Forest")
plt.show()
