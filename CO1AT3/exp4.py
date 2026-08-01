import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
W = np.zeros(X.shape[1])
b = 0
learning_rate = 0.1
epochs = 20

for epoch in range(epochs):
    for i in range(len(X)):
        y_pred = 1 if (np.dot(X[i], W) + b) >= 0 else 0
        error = y[i] - y_pred
        W += learning_rate * error * X[i]
        b += learning_rate * error

print("Weights:", W)
print("Bias:", b)

for i in range(len(X)):
    pred = 1 if (np.dot(X[i], W) + b) >= 0 else 0
    print(f"Input: {X[i]} Prediction: {pred}")