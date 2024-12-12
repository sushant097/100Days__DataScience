import numpy as np
import matplotlib.pyplot as plt

# Simulated Data
np.random.seed(42)
X = np.random.rand(100, 2) # 100 samples, 2 features
y = 3 * X[:, 0] + 5 * X[:, 1]+ np.random.randn(100) * 0.5 # Linear relattion with noise


# Batch Gradient Descent
def batch_gradient_descent(X, y, lr=0.1, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for epoch in range(epochs):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
        loss = np.mean((X.dot(theta) - y) ** 2)
        losses.append(loss)
    return theta, losses

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, lr=0.1, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= lr * gradient
            total_loss += (xi.dot(theta) - yi) ** 2
        losses.append(total_loss / m)
    return theta, losses


# Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, y, lr=0.1, epochs=100, batch_size=10):
    m, n = X.shape
    theta = np.zeros(n)
    losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        total_loss = 0
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            gradient = (1/batch_size) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta -= lr * gradient
            total_loss += np.sum((X_batch.dot(theta) - y_batch) ** 2)
        losses.append(total_loss / m)
    return theta, losses


# Run all methods
_, losses_batch = batch_gradient_descent(X, y)
_, losses_sgd = stochastic_gradient_descent(X, y)
_, losses_mini_batch = mini_batch_gradient_descent(X, y)

# Adjusting the plot to zoom in on fluctuations and improving visibility
plt.figure(figsize=(12, 8))

# Focus on the first 50 epochs for better visualization of fluctuations
epochs_to_display = 50

# Plot convergence
plt.plot(losses_batch[:epochs_to_display], label="Batch Gradient Descent", linewidth=2)
plt.plot(losses_sgd[:epochs_to_display], label="Stochastic Gradient Descent", linestyle='--', linewidth=2)
plt.plot(losses_mini_batch[:epochs_to_display], label="Mini-Batch Gradient Descent", linestyle='-.', linewidth=2)

plt.xlabel("Epochs (First 50)")
plt.ylabel("Mean Squared Error (Loss)")
plt.title("Convergence of Gradient Descent Methods (Zoomed-In View)")
plt.legend()
plt.grid(alpha=0.5)
plt.show()