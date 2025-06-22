# Amy Wang
# ITP 449
# Final Project
# Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1–2. Read dataset and define feature/target sets
df = pd.read_csv("/Users/meinelieben/Desktop/Spring2025/ITP 499/A_Z_Handwritten_Data.csv")
X = df.drop('label', axis=1)
y = df['label']

# 3. Map 0–25 to 'A'–'Z'
word_dict = {i: chr(65 + i) for i in range(26)}
y = y.map(word_dict)

# 4. Print shapes
print("Feature set shape:", X.shape)
print("Target shape:", y.shape)

# 5. Countplot of letter frequencies
plt.figure(figsize=(14, 6))
sns.countplot(x=y, order=sorted(y.unique()), palette="husl")
plt.title("Class Distribution of Letters")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2025, stratify=y
)

# 7. Normalize/Scale feature values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 8. Define MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    max_iter=25,
    alpha=0.001,
    learning_rate_init=0.01,
    random_state=2025,
    verbose=True
)

# 9. Train model
model.fit(X_train, y_train)

# 10. Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("Cross Entropy Loss During Training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 11. Display test accuracy
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
train_acc = model.score(X_train, y_train)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train Accuracy: {train_acc:.4f}")

# 12. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap="viridis")
plt.title("Confusion Matrix (A–Z Letters)")
plt.gcf().set_size_inches(14, 14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 13. Predict the first test image and show actual/predicted letter
first_image = X_test.iloc[0].values.reshape(28, 28)
actual_letter = y_test.iloc[0]
predicted_letter = model.predict(X_test.iloc[[0]])[0]

plt.imshow(first_image, cmap="gray")
plt.title(f"Predicted: ['{predicted_letter}']  Actual: ['{actual_letter}']")
plt.axis("off")
plt.show()

# 14. Display one misclassified image
for i in range(len(y_test)):
    if y_test.iloc[i] != y_pred[i]:
        wrong_idx = i
        break

misclassified_image = X_test.iloc[wrong_idx].values.reshape(28, 28)
plt.imshow(misclassified_image, cmap="gray")
plt.title(f"Predicted: ['{y_pred[wrong_idx]}']  Actual: ['{y_test.iloc[wrong_idx]}']")
plt.axis("off")
plt.show()
