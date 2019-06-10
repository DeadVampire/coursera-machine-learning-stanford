# -*- coding: utf-8 -*-
from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

data3 = loadmat("ex6data3.mat")
X = data3["X"]
y = data3["y"]

pos = X[np.where(y == 1)[0]]
neg = X[np.where(y == 0)[0]]
plt.scatter(pos[:, 0], pos[:, 1], marker = '+')
plt.scatter(neg[:, 0], neg[:, 1], marker = 'o')

c = 1
clf = SVC(C = c, kernel = 'rbf', gamma = 100)
clf.fit(X, y.flatten())

xx, yy = np.meshgrid(np.arange(-0.6, 0.2, 0.02), np.arange(-0.6, 0.6, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, 0)

accuracy = (clf.predict(X) == y.flatten()).sum()/2.11
print("Accuraci is: ", accuracy, "%")