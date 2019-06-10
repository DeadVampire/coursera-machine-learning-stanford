# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data1 = loadmat("ex6data1.mat")
X = data1["X"]
y = data1["y"]

pos = X[np.where(y == 1)[0]]
neg = X[np.where(y == 0)[0]]
plt.scatter(pos[:, 0], pos[:, 1], marker = '+')
plt.scatter(neg[:, 0], neg[:, 1], marker = 'o')

C = 1
clf = SVC(C, kernel = 'linear')
clf.fit(X, y.flatten())

xx, yy = np.meshgrid(np.arange(0, 4.5, 0.02), np.arange(0, 4.5, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, 0)