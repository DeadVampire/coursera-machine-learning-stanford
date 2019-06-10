# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
from sigmoid import sigmoid

data = loadmat("ex3data1.mat")
X = data["X"]
X = np.hstack((np.ones((5000, 1)), X))
y = data["y"]
y[y == 10] = 0
weights = loadmat("ex3weights.mat")
theta1 = weights["Theta1"]
theta2 = weights["Theta2"]

layer1 = sigmoid(X@theta1.T)
layer1 = np.hstack((np.ones((5000, 1)), layer1))

output = sigmoid(layer1@theta2.T)

output = np.argmax(output, axis = 1) + 1
output[output == 10] = 0

accuracy = ((output.reshape((5000, 1)) == y).sum())/50
print("Accuracy: ", accuracy, "%")