import matplotlib.pyplot as plt
import numpy as np

my_data = np.genfromtxt('data.csv', delimiter= ',')

def compute_cost(X, y, theta):
    variance = np.power((X @ theta.T - y), 2)
    return np.sum(variance) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iteration):
    for i in range(iteration):
        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
        cost = compute_cost(X, y, theta)
    
    return (theta, cost)

alpha = 0.0001
"learning rate"

iteration = 1000

X = my_data[:, 0].reshape(-1, 1)

"""\'reshape' means to convert
the array into a matrix, reshape(rows, column)

-1 indicates that python will determine the number of rows itself
"""

ones = np.ones([X.shape[0], 1])
X = np.concatenate([ones, X], 1)

theta = np.array([[1.0, 1.0]])

y = my_data[:, 1].reshape(-1, 1)

g, cost = gradientDescent(X, y, theta, alpha, iteration)
print(g, cost)

plt.scatter(my_data[:,0].reshape(-1, 1), y)
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = g[0][0] + g[0][1]* x_vals

plt.plot(x_vals, y_vals, '--')

plt.show()
