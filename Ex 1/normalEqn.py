# -*- coding: utf-8 -*-
import numpy as np

data = np.loadtxt("ex1data2.txt", delimiter = ",")
m, n = data.shape

X = data[:, :n-1]
X = np.hstack((np.ones((m, 1)), X))
y = data[:, n-1:]

theta = (np.linalg.pinv(X.T@X))@(X.T@y)

print(theta[0] + (theta[1] * 1650) + (theta[2] * 3))