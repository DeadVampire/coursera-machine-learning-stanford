# -*- coding: utf-8 -*-
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sigmoid import sigmoid


data = loadmat("ex4data1.mat")
X = data["X"]
X = np.hstack((np.ones((5000, 1)), X))
y = data["y"]
yk = OneHotEncoder(sparse = False).fit_transform(y)

weights = loadmat("ex4weights.mat")
theta1 = weights["Theta1"]
theta2 = weights["Theta2"]


def feed_forward(theta1, theta2, X):
    m = len(X)
    z2 = X @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)
    return z2, a2, z3, a3


def cost(theta1, theta2, X, y, l = 0):
    m = len(y)
    predict = feed_forward(theta1, theta2, X)[-1]
    J = y * np.log(predict) + (1 - y) * np.log(1 - predict)
    J_reg = (theta1[:, 1:]**2).sum() + (theta2[:, 1:]**2).sum()
    
    cost = -J.sum()/m + (l/(2 * m)) * J_reg
    return cost


print("Cost with pre-trained weights: ", cost(theta1, theta2, X, yk))
print("Regularized cost with pre-trained weights: ", cost(theta1, theta2, X, yk, l = 1))


def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def grad(theta1, theta2, X, y):
    m = len(X)
    l = 1
    grad1 = np.zeros_like(theta1)
    grad2 = np.zeros_like(theta2)
    z2, a2, z3, a3 = feed_forward(theta1, theta2, X)
    delta3 = a3 - y
    delta2 = (delta3@theta2)[:, 1:] * sigmoid_grad(z2)
    grad1 = (grad1 + delta2.T@X)/m
    grad1[:, 1:] += (l/m) * theta1[:, 1:]
    grad2 = (grad2 + delta3.T@a2)/m
    grad2[:, 1:] += (l/m) * theta2[:, 1:]
    return grad1, grad2

def grad_descent(theta1, theta2, X, y, alpha):
    grad1, grad2 = grad(theta1, theta2, X, y)
    theta1 -= alpha * grad1
    theta2 -= alpha * grad2
    return theta1, theta2
    
def train_nn(iters, X, y, alpha):
    theta1 = np.random.uniform(-0.12, 0.12, (25, 401))
    theta2 = np.random.uniform(-0.12, 0.12, (10, 26))
    
    for i in range(iters):
        theta1, theta2 = grad_descent(theta1, theta2, X, y, alpha)
    return theta1, theta2

alpha = 2
theta_l1, theta_l2 = train_nn(400, X, yk, alpha)

predict = feed_forward(theta_l1, theta_l2, X)[-1]
predict = np.argmax(predict, axis = 1) + 1
accuracy = (predict == y.flatten()).sum()/50

print("Accuracy of our trained model: ", accuracy, "%")