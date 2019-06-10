from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


data = loadmat("ex5data1.mat")

X = data['X']
X = np.hstack((np.ones((len(X), 1)), X))
y = data['y']

X_test = data['Xtest']
X_test = np.hstack((np.ones((len(X_test), 1)), X_test))
y_test = data['ytest']

X_val = data['Xval']
X_val = np.hstack((np.ones((len(X_val), 1)), X_val))
y_val = data['yval']

l = 1


def cost(theta, X, y):
    m = len(X)
    theta = theta.reshape((len(theta), 1))
    hx = X@theta
    J = ((hx - y)**2).sum()/(2 * m)
    J += (l/(2 * m)) * (theta[1:]**2).sum()
    return J


def grad(theta, X, y):
    theta = theta.reshape((len(theta), 1))
    m = len(X)
    hx = X@theta
    grad = ((hx - y) * X).sum(axis = 0)/m
    grad[1:] += (l/m) * theta[1:, 0]
    return grad


def trainLinearReg(X, y):
    theta = np.ones(X.shape[1])
    min = minimize(cost, theta, (X, y), "CG", jac = grad)
    print("Function minimized: ", min['success'])
    theta = min['x'].reshape((len(theta), 1))
    return theta


def learningCurve(X, y, X_val, y_val):
    error_train = np.zeros(len(X) - 1)
    error_val = np.zeros(len(X) - 1)
    
    for i in range(1, len(X)):
        theta = trainLinearReg(X[0:i+1, :], y[0:i+1, :])
        error_train[i - 1] = cost(theta, X[0:i+1, :], y[0:i+1, :])
        error_val[i - 1] = cost(theta, X_val, y_val)
        
    plt.plot(range(2, 13), error_train)
    plt.plot(range(2, 13), error_val)
    plt.show()


def polyFeatures(X, p):
    a = PolynomialFeatures(degree = p)
    X = a.fit_transform(X)
    return X


def featureNormalize(X):
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    X = (X - mean)/std
    return X, mean, std



##Linear Regressiom
theta = np.ones(X.shape[1])
theta = trainLinearReg(X, y)

plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], X@theta)
plt.show()

##Linear Regressiom learning curve
learningCurve(X, y, X_val, y_val)



##Polynomial Regression
p = 8
X_poly = polyFeatures(X[:, 1:], p)
X_poly[:, 1:], mean, std = featureNormalize(X_poly[:, 1:])

theta_poly = trainLinearReg(X_poly, y)

xp = np.arange(-50, 50, 10).reshape((10, 1))
xp1 = polyFeatures(xp, p)
xp1[:, 1:] = (xp1[:, 1:] - mean)/std

predict = xp1@theta_poly
plt.scatter(X[:, 1], y)
plt.plot(xp, predict)
plt.show()

#Polynomial Regressiom learning curve
X_val = polyFeatures(X_val[:, 1:], p)
X_val[:, 1:] = (X_val[:, 1:] - mean)/std

learningCurve(X_poly, y, X_val, y_val)

X_test = polyFeatures(X_test[:, 1:], p)
X_test[:, 1:] = (X_test[:, 1:] - mean)/std
print(cost(theta_poly, X_test, y_test))