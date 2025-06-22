# Amy Wang
# ITP 449
# Final Project
# Question 3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Read A_Z_Handwritten_Data.csv into a DataFrame
df = pd.read_csv("/Users/meinelieben/Desktop/Spring2025/ITP 499/A_Z_Handwritten_Data.csv")

# 2. Define features and target, and map 0–25 to 'A'–'Z'
X_az = df.drop('label', axis=1)
y_az = df['label']
letter_dict = {i: chr(i + 65) for i in range(26)}
y_az = y_az.map(letter_dict)

# 3. Train Test Split
X_trainLet, X_testLet, y_trainLet, y_testLet = train_test_split(
    X_az, y_az, test_size=1/7, random_state=2025, stratify=y_az
)
# 4. Scale and Normalize features
X_trainLet = X_trainLet / 255.0
X_testLet = X_testLet / 255.0

# 5. Read mnist_train.csv and mnist_test.csv
mnist_train = pd.read_csv("/Users/meinelieben/Desktop/Spring2025/ITP 499/mnist_train.csv")
mnist_test = pd.read_csv("/Users/meinelieben/Desktop/Spring2025/ITP 499/mnist_test.csv")

# 6. Define digit feature and label sets, map digits to string characters
X_trainNum = mnist_train.drop('label', axis=1)
y_trainNum = mnist_train['label']
X_testNum = mnist_test.drop('label', axis=1)
y_testNum = mnist_test['label']

digit_dict = {i: str(i) for i in range(10)}
y_trainNum = y_trainNum.map(digit_dict)
y_testNum = y_testNum.map(digit_dict)

# 7. Scale digit features
X_trainNum = X_trainNum / 255.0
X_testNum = X_testNum / 255.0

# 8. Concatenate letter and digit datasets
X_train = pd.concat([X_trainLet, X_trainNum], ignore_index=True)
X_test = pd.concat([X_testLet, X_testNum], ignore_index=True)
y_train = pd.concat([y_trainLet, y_trainNum], ignore_index=True)
y_test = pd.concat([y_testLet, y_testNum], ignore_index=True)

# 9. Show a histogram of class distribution in training set
plt.figure(figsize=(14, 6))
sns.countplot(x=y_train, order=sorted(y_train.unique()), palette="husl" )
plt.title("Class Distribution in Training Set (Letters + Digits)")
plt.xlabel("Labels")
plt.ylabel("Count")
plt.show()

# 10. Create an MLPClassifier
model = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    max_iter=100,
    alpha=0.001,
    learning_rate_init=0.01,
    random_state=2025,
    verbose=True # this one to show progress per iteration!
)

# 11. Fit the model to training data
model.fit(X_train, y_train)

# 12. Plot the loss curve
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("Cross Entropy Loss During Training")
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy Loss")
plt.grid(True)
plt.show()

# 13. Accuracy score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# 14. Plot confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, cmap='viridis', colorbar=True, xticks_rotation='vertical')
plt.title("Confusion Matrix: Letters and Digits")
plt.show()


# 15. Read phrase in image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
# Load and convert the image to grayscale
img = mpimg.imread('/Users/meinelieben/Desktop/Spring2025/ITP 499/testPhrase.png')
r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]
imageData = 0.299 * r + 0.587 * g + 0.114 * b

# Predict each 28x28 slice of the phrase
print("Image shape:", imageData.shape)

phrasePred = ""
for i in range(6):
    sample = imageData[:, 28*i : 28*(i+1)]  # Slice one character
    sampleData = sample.reshape(1, -1)  # Flatten to 1D
    sampleDF = pd.DataFrame(sampleData, columns=X_train.columns)  # Match feature columns
    modelPred = model.predict(sampleDF)
    phrasePred += str(modelPred[0])

# Show result
plt.imshow(imageData, cmap='gray')
plt.title("Model Prediction: " + phrasePred)
plt.axis('off')
plt.show()
