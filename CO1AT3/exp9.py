import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
W1 = np.random.uniform(size=(2, 2))
W2 = np.random.uniform(size=(2, 1))

for epoch in range(10000):
    layer1 = sigmoid(np.dot(X, W1))
    layer2 = sigmoid(np.dot(layer1, W2))
    
    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_deriv(layer2)
    
    layer1_error = layer2_delta.dot(W2.T)
    layer1_delta = layer1_error * sigmoid_deriv(layer1)
    
    W2 += layer1.T.dot(layer2_delta)
    W1 += X.T.dot(layer1_delta)

print("Final Predictions:\n", layer2)