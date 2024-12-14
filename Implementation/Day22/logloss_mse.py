# Re-importing libraries after environment reset
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for visualization
y_true_class = np.array([1, 0, 1, 1])  # Binary classification ground truth
y_pred_probs = np.linspace(0.01, 0.99, 100)  # Predicted probabilities

y_true_reg = np.array([3.5, -0.5, 2.0, 7.0])  # Regression ground truth
y_pred_vals = np.linspace(-1, 8, 100)  # Predicted values

# Cross-Entropy Loss calculation
cross_entropy_losses = -(
    y_true_class[0] * np.log(y_pred_probs) +
    (1 - y_true_class[0]) * np.log(1 - y_pred_probs)
)

# Mean Squared Error Loss calculation
mse_losses = (y_pred_vals - y_true_reg[0])**2

# Visualization
plt.figure(figsize=(14, 6))

# Cross-Entropy Loss
plt.subplot(1, 2, 1)
plt.plot(y_pred_probs, cross_entropy_losses, label="Cross-Entropy Loss", color="blue")
plt.axvline(x=1, linestyle="--", color="gray", label="Correct Class (1)")
plt.axvline(x=0, linestyle="--", color="gray", label="Correct Class (0)")
plt.xlabel("Predicted Probability")
plt.ylabel("Loss")
plt.title("Cross-Entropy Loss Behavior")
plt.legend()
plt.grid(alpha=0.5)

# Mean Squared Error Loss
plt.subplot(1, 2, 2)
plt.plot(y_pred_vals, mse_losses, label="MSE Loss", color="green")
plt.axvline(x=y_true_reg[0], linestyle="--", color="red", label="Ground Truth")
plt.xlabel("Predicted Value")
plt.ylabel("Loss")
plt.title("Mean Squared Error Loss Behavior")
plt.legend()
plt.grid(alpha=0.5)

plt.tight_layout()
plt.show()
