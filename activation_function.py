import numpy as np
import matplotlib.pyplot as plt

# Define 4 activation functions
def sigmoidFunction(x):
    return 1 / (1 + np.exp(-x))

def reluFunction(x):
    return np.maximum(0, x)

def leaky_reluFunction(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def tanhFunction(x):
    return np.tanh(x)


# x = np.linspace(-5, 5, 100)
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
# finding the y values 
y_sigmoid = sigmoidFunction(x)
# y_relu = reluFunction(x)
# y_leaky_relu = leaky_reluFunction(x)
# y_tanh = tanhFunction(x)

# Plot the graph for activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoid')
# plt.plot(x, y_relu, label='ReLU')
# plt.plot(x, y_leaky_relu, label='Leaky ReLU')
# plt.plot(x, y_tanh, label='Tanh')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
