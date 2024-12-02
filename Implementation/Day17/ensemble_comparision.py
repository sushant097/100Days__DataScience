import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest (Bagging)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
rf_loss = log_loss(y_test, rf_pred)

# Train AdaBoost (Boosting)
ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_model.fit(X_train, y_train)
ab_pred = ab_model.predict_proba(X_test)[:, 1]
ab_acc = accuracy_score(y_test, ab_model.predict(X_test))
ab_loss = log_loss(y_test, ab_pred)

# Combine metrics
models = ["Random Forest (Bagging)", "AdaBoost (Boosting)"]
accuracy = [rf_acc, ab_acc]
loss = [rf_loss, ab_loss]

# Determine dynamic xlim for Log Loss
log_loss_max = max(loss)
log_loss_margin = log_loss_max * 0.1  # 10% padding for better visibility

# Improved Visualization
fig, ax1 = plt.subplots(figsize=(12, 6))

# Horizontal Bar for Accuracy
ax1.barh(models, accuracy, color='dodgerblue', label='Accuracy', height=0.4, align='center', alpha=0.8)
ax1.set_xlabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax1.set_xlim(0.8, 1.0)  # Focus on relevant range for accuracy
ax1.set_title("Model Performance Comparison: Accuracy and Log Loss", fontsize=14, fontweight='bold', pad=20)

# Add gridlines for accuracy axis
ax1.grid(axis='x', linestyle='--', alpha=0.6)

# Line Plot for Log Loss
ax2 = ax1.twiny()
ax2.plot(loss, models, color='crimson', marker='o', label='Log Loss', linestyle='--', linewidth=2)
ax2.set_xlabel("Log Loss", fontsize=12, fontweight='bold')
ax2.set_xlim(0.0, log_loss_max + log_loss_margin)  # Dynamically adjust range

# Annotate exact values for accuracy and log loss
for i, (acc, ls) in enumerate(zip(accuracy, loss)):
    ax1.text(acc - 0.015, i, f"{acc*100:.1f}%", color="black", va='center', fontsize=10, fontweight='bold')
    ax2.text(ls + log_loss_margin / 5, i, f"{ls:.3f}", color="black", va='center', fontsize=10, fontweight='bold')

# Add a description footer
plt.figtext(0.5, -0.05, "Accuracy: Higher is better | Log Loss: Lower is better", fontsize=10, ha="center", color="gray", fontstyle='italic')

# Add subtle background grid for log loss
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# Legends
ax1.legend(["Accuracy"], loc="lower left", bbox_to_anchor=(0.1, -0.15), fontsize=10, frameon=False)
ax2.legend(["Log Loss"], loc="lower right", bbox_to_anchor=(0.9, -0.15), fontsize=10, frameon=False)

plt.tight_layout()
plt.show()
