# -*- coding: utf-8 -*-
from sigmoid import sigmoid
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data = np.loadtxt("ex2data2.txt", delimiter = ",")
m, n = data.shape
X = data[:, :n-1]
X = PolynomialFeatures(6).fit_transform(X)
y = data[:, n-1:]
theta = np.zeros((28, 1))
l = 1

def cost(theta, X, y):
    theta = theta.reshape((28, 1))
    hypo = sigmoid(X@theta)
    cost = (((-y * np.log(hypo)) - ((1 - y) * np.log(1 - hypo))).sum(axis = 0))/m
    cost += (l/(2 * m)) * ((theta[1:, :]**2).sum(axis = 0))
    return cost[0]

def grad(theta, X, y):
    theta = theta.reshape((28, 1))
    hypo = sigmoid(X@theta)
    grad = (((hypo - y) * X).sum(axis = 0))/m
    grad = grad.reshape((28, 1))
    grad[1:, :] += (l/m) * theta[1:, :]
    grad = grad.reshape((28,))
    return grad

def predict(X, theta):
    predict = sigmoid(X@theta)
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    return predict

def decision_boundary(X, y, theta):
    pos = X[np.where(y==1)[0]]
    neg = X[np.where(y==0)[0]]
    plt.scatter(pos[:, 1], pos[:, 2], marker = '+')
    plt.scatter(neg[:, 1], neg[:, 2], marker = 'o')
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            a = np.asarray((u[i], v[j])).reshape((1, 2))
            z[i, j] = (PolynomialFeatures(6).fit_transform(a))@theta
    plt.contour(u, v, z.T, 0)
    plt.legend(loc = 0)
    
min = minimize(cost, theta, (X, y), method = "CG", jac = grad)
theta = min['x'].reshape((28, 1))

accuracy = ((predict(X, theta) == y).sum()/m) * 100
print("Accuracy of training model is:", accuracy, "%")
decision_boundary(X, y, theta)