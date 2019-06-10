# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


data = loadmat("ex8data1.mat")
X = data['X']
Xval = data['Xval']
yval = data['yval']

plt.scatter(X[:, 0], X[:, 1])


def gaussParams(X):
    mu = np.mean(X, axis = 0, keepdims = True)
    #cov = ((X - mu)**2).sum(axis = 0)/len(X)
    cov = ((X-mu).T@(X- mu))/len(X)
    return mu, cov

def estimateGaussian(X, cov, mu):
    k = len(cov)
    
    X = X - mu
    
    detCov = np.linalg.det(cov)
    
    p = (1/(2 * np.pi))**(k/2)
    p = p * (detCov ** -0.5)
    p = p * np.exp((-0.5 * (X @ np.linalg.pinv(cov)) * X).sum(axis = 1))
    
    return p

def selectThreshold(Xval, yval, cov, mu):
    gaussXval = estimateGaussian(Xval, cov, mu)
    
    maxf1 = 0
    bestEps = 0

    evals = np.linspace(np.min(gaussXval), np.max(gaussXval), 1000)
    
    for e in evals:
        pred = (gaussXval < e).reshape((-1, 1))
        tp = (pred & yval).sum()
        fp = (yval[np.where(pred == 1)[0]] == 0).sum()
        fn = (pred[np.where(yval == 1)[0]] == 0).sum()
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        f1 = (2 * prec * rec)/(prec + rec)
        
        if f1 > maxf1:
            maxf1 = f1
            bestEps = e
        
    return bestEps


mu, cov = gaussParams(X)

gaussX = estimateGaussian(X, cov, mu)

meshVal = np.arange(0, 30, 0.1)
xx, yy = np.meshgrid(meshVal, meshVal)
zz = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1, 1))))
zz = estimateGaussian(zz, cov, mu).reshape(300, 300)

mylevels = np.array([10.0**i for i in np.arange(-20,0,3)])
plt.contour(xx, yy, zz, mylevels)

e = selectThreshold(Xval, yval, cov, mu)

anomalies = X[np.where(gaussX < e)]
plt.scatter(anomalies[:, 0], anomalies[:, 1], marker = 'x')
plt.show()



data2 = loadmat("ex8data2.mat")
X2 = data2['X']
Xval2 = data2['Xval']
yval2 = data2['yval']

mu2, cov2 = gaussParams(X2)

gaussX2 = estimateGaussian(X2, cov2, mu2)

e2 = selectThreshold(Xval2, yval2, cov2, mu2)

print("Value of e for data2 is: ", e2)

anomalies2 = X2[np.where(gaussX2 < e2)]
print("Number of anomalies: ", len(anomalies2))