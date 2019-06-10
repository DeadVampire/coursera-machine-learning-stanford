# -*- coding: utf-8 -*-
from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

def gaussian_fun(x1, x2, sigma):
    a = -(((x1 - x2)**2).sum()/(2 * (sigma**2)))
    return np.exp(a)
#
#def gaussian_kernel(X, y):
#    sigma = 0.1
#    m = X.shape[0]
#    result = np.zeros((m, m))
#    for i in range(m):
#        for j in range(m):
#            result[i][j] = gaussian_fun(X[i], X[j], sigma)
#    return result

x1 = np.asarray([1, 2, 1])
x2 = np.asarray([0, 4, -1])
sigma = 2

sim = gaussian_fun(x1, x2, sigma)
print("Gaussian Kernel function returned: ", sim)

data2 = loadmat("ex6data2.mat")
X = data2["X"]
y = data2["y"]

pos = X[np.where(y == 1)[0]]
neg = X[np.where(y == 0)[0]]
plt.scatter(pos[:, 0], pos[:, 1], marker = '+')
plt.scatter(neg[:, 0], neg[:, 1], marker = 'o')

c = 1
clf = SVC(C = c, kernel = 'rbf', gamma = 100)
clf.fit(X, y.flatten())

xx, yy = np.meshgrid(np.arange(0, 1, 0.02), np.arange(0.4, 1, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, 0)
    