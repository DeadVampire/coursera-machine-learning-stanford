# -*- coding: utf-8 -*-
from scipy.io import loadmat
from sigmoid import sigmoid
import numpy as np
from scipy.optimize import minimize

data = loadmat("ex3data1.mat")
X = data["X"]
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
y = data["y"]
k = 10
y[y == 10] = 0
theta = np.zeros((n+1, k))
l = 2

def cost(theta, X, y):
    theta = theta.reshape((n+1, 1))
    hypo = sigmoid(X@theta)
    cost = (((-y * np.log(hypo)) - ((1 - y) * np.log(1 - hypo))).sum(axis = 0))/m
    cost += (l/(2 * m)) * ((theta[1:, :]**2).sum(axis = 0))
    return cost

def grad(theta, X, y):
    theta = theta.reshape((n+1, 1))
    hypo = sigmoid(X@theta)
    grad = (((hypo - y) * X).sum(axis = 0))/m
    grad = grad.reshape((401, 1))
    grad[1:, :] += (l/m) * theta[1:, :]
    grad = grad.reshape((401,))
    return grad

for i in range(k):
    y_k = np.where(y == i, 1, 0)
    min = minimize(cost, theta[:, i:i+1], (X, y_k), method = "BFGS", jac = grad)
    theta[:, i] = min["x"]
    
predict = sigmoid(X@theta)
predict = np.argmax(predict, axis = 1)
accuracy = (predict.reshape((5000, 1)) == y).sum()/50
print("Accuracy is: ", accuracy, "%")