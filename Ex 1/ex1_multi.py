# -*- coding: utf-8 -*-
import numpy as np

data = np.loadtxt("ex1data2.txt", delimiter = ",")
m, n = data.shape

X = data[:, :n-1]
X_mean = np.mean(X, axis = 0).reshape((1, 2))
X_std = np.std(X, axis = 0).reshape((1, 2))
X = (X - X_mean)/X_std
X = np.hstack((np.ones((m, 1)), X))
y = data[:, n-1:]

theta = np.zeros((n, 1))
alpha = 0.01
iterations = 1500

def cost(X, y, theta):
    hypo = X@theta
    cost = (((hypo - y)**2).sum(axis = 0))/(2*m)
    return cost
    
def gradient(X, y, theta, alpha):
    hypo = X@theta
    grads = ((alpha/m) * ((hypo - y) * X)).sum(axis = 0)
    return grads

def grad_descent(X, y, theta, alpha):
    grads = gradient(X, y, theta, alpha)
    for i in range(n):
        theta[i] = theta[i] - grads[i]
    return theta

def predict(a):
    predict = theta[0]
    for i in range(n-1):
        predict += theta[i+1] * ((a[i] - X_mean[0, i])/X_std[0, i])
    return predict

for i in range(iterations):
    theta = grad_descent(X, y, theta, alpha)
    
print(predict((1650, 3)))
    
    
