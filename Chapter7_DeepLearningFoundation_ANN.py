import matplotlib_inline
import numpy as np
import matplotlib.pyplot as plt

x = np.array([ [0, 1], [1, 0], [1, 1], [0, 0] ])
y = np.array([ [1], [1], [0], [0] ])

num_input = 2
num_hidden = 5
num_output = 1

Wxh = np.random.rand(num_input, num_hidden)
bh = np.zeros((1, num_hidden))

Why = np.random.rand(num_hidden, num_output)
by = np.zeros((1, num_output))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

def forward_prop(x, Wxh, Why):
    z1 = np.dot(x, Wxh) + bh
    a1 = sigmoid(z1)
    z2 = np.dot(a1, Why) + by
    y_hat = sigmoid(z2)
    return z1, a1, z2, y_hat

def backward_prop(y_hat ,z1, a1, z2):
    # Get the gradient of the cost function with respect to W_hy
    delta2 = np.multiply(-(y - y_hat), sigmoid_derivative(z2))
    dJ_dWhy = np.dot(a1.T, delta2)
    # Get the gradient of the cost function with respect to W_xh
    delta1 = np.dot(delta2, Why.T) * sigmoid_derivative(z1)
    dJ_dWxh = np.dot(x.T, delta1)
    return dJ_dWxh, dJ_dWhy

def cost_function(y, y_hat):
    J = 0.5 * sum((y - y_hat) ** 2)
    return J

alpha = 0.01            # Learning rate
num_iterations = 5000

cost = []

for i in range(num_iterations):
    z1, a1, z2, y_hat = forward_prop(x, Wxh, Why)
    dJ_dWxh, dJ_dWhy = backward_prop(y_hat, z1, a1, z2)

    # Update Weights
    Wxh = Wxh - alpha * dJ_dWxh
    Why = Why - alpha * dJ_dWhy

    # Compute cost
    c = cost_function(y, y_hat)

    cost.append(c)

plt.grid()
plt.plot(range(num_iterations), cost)

plt.title('Cost Function')
plt.xlabel('Training Iterations')
plt.ylabel('Cost')
plt.show()