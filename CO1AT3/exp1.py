import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def artificial_neuron(inputs, weights, bias):
    weighted_sum = np.dot(inputs, weights) + bias
    return step_function(weighted_sum)

inputs = np.array([1.5, 2.0, -1.0])
weights = np.array([0.5, -0.2, 0.8])
bias = -0.5

output = artificial_neuron(inputs, weights, bias)
print("Output:", output)