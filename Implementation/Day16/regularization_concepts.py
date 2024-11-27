import numpy as np
import matplotlib.pyplot as plt

# Define the loss function (ellipses)
def loss_function(theta1, theta2):
    return theta1**2 + 4 * theta2**2

# Generate a grid for the loss function contours
theta1 = np.linspace(-2, 2, 100)
theta2 = np.linspace(-2, 2, 100)
Theta1, Theta2 = np.meshgrid(theta1, theta2)
Z = loss_function(Theta1, Theta2)

# Plot the L1, L2, and Elastic Net constraint regions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# L1 Regularization
axes[0].contour(Theta1, Theta2, Z, levels=10, linewidths=0.5, colors="blue")
axes[0].plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="orange", label="L1 Norm")
axes[0].set_title("L1 Regularization (Sparsity Inducing)")
axes[0].set_xlabel("Theta1")
axes[0].set_ylabel("Theta2")
axes[0].legend()

# L2 Regularization
axes[1].contour(Theta1, Theta2, Z, levels=10, linewidths=0.5, colors="blue")
circle = plt.Circle((0, 0), 1, color="orange", fill=False, label="L2 Norm")
axes[1].add_artist(circle)
axes[1].set_title("L2 Regularization (Weight Sharing)")
axes[1].set_xlabel("Theta1")
axes[1].set_ylabel("Theta2")
axes[1].legend()

# Elastic Net Regularization
axes[2].contour(Theta1, Theta2, Z, levels=10, linewidths=0.5, colors="blue")
axes[2].plot([-0.7, 0.7, 0.7, -0.7, -0.7], [-1, -1, 1, 1, -1], color="purple", linestyle="--", label="L1 Component")
circle_en = plt.Circle((0, 0), 0.7, color="orange", fill=False, label="L2 Component")
axes[2].add_artist(circle_en)
axes[2].set_title("Elastic Net Regularization (Compromise)")
axes[2].set_xlabel("Theta1")
axes[2].set_ylabel("Theta2")
axes[2].legend()

plt.tight_layout()
plt.show()
