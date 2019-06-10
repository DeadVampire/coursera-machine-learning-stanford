# -*- coding: utf-8 -*-
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


data = loadmat("ex7data1.mat")
X = data['X']


def featureNormalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    return (X - mean)/std, mean, std

def pca(X):
    m = len(X)
    sigma = (X.T@X)/m
    U, S, V = np.linalg.svd(sigma)
    return U

def projectData(X, U, K):
    U_reduce = U[:, :K]
    Z = X@U_reduce
    return Z

def recoverData(Z, U):
    U = U[:, :Z.shape[1]]
    return Z@U.T


K = 1
X = featureNormalize(X)[0]

plt.scatter(X[:, 0], X[:, 1], marker = 'o')

U = pca(X)
Z = projectData(X, U, K)

X_rcvr = recoverData(Z, U)
plt.scatter(X_rcvr[:, 0], X_rcvr[:, 1], c = "Red")
plt.show()


faces = loadmat("ex7faces.mat")
faces = faces['X']
plt.imshow(faces[0, :].reshape((32, 32)))
plt.show()
faces = featureNormalize(faces)[0]

K = 100

U_faces = pca(faces)
Z_faces = projectData(faces, U_faces, K)
faces_rcvr = recoverData(Z_faces, U_faces)

plt.imshow(faces_rcvr[0, :].reshape((32, 32)))
plt.show()