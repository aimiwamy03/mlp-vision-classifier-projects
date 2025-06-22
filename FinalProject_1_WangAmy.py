# Amy Wang
# ITP 449
# Final Project
# Question 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier


# Step 1: Set seed for reproducibility
np.random.seed(2025)

# Number of samples
n = 1000

# Generate theta: uniformly random angles from 0 to 2π
theta = np.random.uniform(0, 2 * np.pi, n)

# Labels: half 0s, half 1s
label = np.random.choice([0, 1], size=n)

# Compute radius with conditional sign flipping
r = (2 * theta + np.pi) * (-1)**label

# Compute x and y, with noise added
x = r * np.cos(theta) + np.random.randn(n)
y = r * np.sin(theta) + np.random.randn(n)

# Create DataFrame
myDF = pd.DataFrame({
    'label': label,
    'X1': x,
    'X2': y
})

# Step 2: generate a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='X1', y='X2', hue='label', palette=['purple', 'yellow'], data=myDF)
plt.title('Spiral Data')
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

# Step 3: Define features and target dataframes.
X = myDF[['X1', 'X2']]
y = myDF['label']

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2025)

# Step 4; Train the network using MLP Classifier from scikit-learn
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.005,
    max_iter=150,
    random_state=2025
)
model.fit(X_train, y_train)

# Step 5: plotting loss curve
plt.plot(model.loss_curve_)
plt.title("MLP Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# Step 6: check accuracy and print it
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.3f}")

# Step 7: Build a confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Spiral 0', 'Spiral 1']).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# Step 8: Plot a decision boundary
# Create a mesh grid
xx, yy = np.meshgrid(np.linspace(-20, 20, 400), np.linspace(-20, 20, 400))
mesh_points = np.c_[xx.ravel(), yy.ravel()]

# Predict classes for each point in the mesh
Z = model.predict(mesh_points).reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.4)
sns.scatterplot(x='X1', y='X2', hue='label', data=myDF, palette='coolwarm', edgecolor='k')
plt.title("Decision Boundary with Spiral Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
