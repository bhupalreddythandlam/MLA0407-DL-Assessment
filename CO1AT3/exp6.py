import numpy as np
import time
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)

def compute_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)

def batch_gd(X, y, lr=0.01, epochs=100):
    m = len(y)
    theta = np.zeros(X.shape[1])
    start = time.time()
    for _ in range(epochs):
        theta -= (lr / m) * X.T.dot(X.dot(theta) - y)
    return time.time() - start, compute_cost(X, y, theta)

def sgd(X, y, lr=0.01, epochs=100):
    m = len(y)
    theta = np.zeros(X.shape[1])
    start = time.time()
    for _ in range(epochs):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            theta -= (lr) * xi.T.dot(xi.dot(theta) - yi).flatten()
    return time.time() - start, compute_cost(X, y, theta)

t_bgd, c_bgd = batch_gd(X, y)
t_sgd, c_sgd = sgd(X, y)

print(f"Batch GD - Time: {t_bgd:.4f}s, Cost: {c_bgd:.4f}")
print(f"SGD - Time: {t_sgd:.4f}s, Cost: {c_sgd:.4f}")