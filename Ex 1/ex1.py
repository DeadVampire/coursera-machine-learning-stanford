import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ex1data1.txt", delimiter = ",")
m, n = data.shape

X = np.hstack((np.ones((m, 1)), data[:, :1]))
y = data[:, 1:]

theta = np.zeros((n, 1))
alpha = 0.01
iterations = 500

plt.scatter(X[:, 1:2], y)

def gradient(X, y, theta, alpha):
    hypo = X@theta
    grads = ((alpha/m) * ((hypo - y) * X)).sum(axis = 0)
    return grads

def grad_descent(X, y, theta, alpha):
    grads = gradient(X, y, theta, alpha)
    for i in range(n):
        theta[i] = theta[i] - grads[i]
    return theta

for i in range(iterations):
    predict = X@theta
    plt.plot(X[:, 1:2], predict)
    theta = grad_descent(X, y, theta, alpha)
