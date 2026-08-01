import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([0.5, 0.8])

W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
b1 = np.array([0.1, 0.2])
W2 = np.array([0.5, 0.6])
b2 = np.array([0.3])

z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)

z2 = np.dot(a1, W2) + b2
output = sigmoid(z2)

print("Hidden Layer Weighted Sum:", z1)
print("Hidden Layer Activation:", a1)
print("Output Layer Weighted Sum:", z2)
print("Final Output:", output)