import matplotlib.pyplot as plt

def cost_function(x):
    return x ** 2 + 5 * x + 6

def gradient(x):
    return 2 * x + 5

x = 10.0
learning_rate = 0.1
iterations = 50
history = []

for i in range(iterations):
    history.append(cost_function(x))
    x = x - learning_rate * gradient(x)

plt.plot(range(iterations), history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Optimization")
plt.show()