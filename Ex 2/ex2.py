# -*- coding: utf-8 -*-
from sigmoid import sigmoid
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.loadtxt("ex2data1.txt", delimiter = ",")
m, n = data.shape
X = data[:, :n-1]
y = data[:, n-1:]
pos = X[np.where(y==1)[0]]
neg = X[np.where(y==0)[0]]
X = np.hstack((np.ones((m, 1)), X))
theta = np.zeros((n, 1))

def cost(theta, X, y):
    theta = theta.reshape((n, 1))
    hypo = sigmoid(X@theta)
    cost = (((-y * np.log(hypo)) - ((1 - y) * np.log(1 - hypo))).sum(axis = 0))/m
    return cost[0]
    
def grad(theta, X, y):
    theta = theta.reshape((n, 1))
    hypo = sigmoid(X@theta)
    grad = ((hypo - y) * X).sum(axis = 0)
    return grad/m

def predict(marks, theta):
    marks = np.asarray(marks).reshape((1, 2))
    predict = theta[0] + marks@(theta[1:, :])
    return sigmoid(predict[0])

ans = minimize(cost, theta, (X, y), method = "Newton-CG", jac = grad)
theta = ans['x'].reshape((n, 1))

plt.scatter(pos[:, 0], pos[:, 1], marker = '+')
plt.scatter(neg[:, 0], neg[:, 1], marker = 'o')

plt.plot(X[:, 1], -((X[:, 1] * theta[1]) + theta[0])/theta[2])

print("Admission probability with Exam 1 : 45 and Exam 2 : 85 is ", predict((45, 85), theta))

    
