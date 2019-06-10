from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


data = loadmat("ex7data2.mat")
X = data['X']

plt.scatter(X[:, 0], X[:, 1], marker = '+')
plt.show()

K = 3


def findClosestCentroids(X, centroids):
    m = len(X)
    K = len(centroids)
    c = np.zeros((m, K))
    for i in range(K):
        c[:, i] = ((X  - centroids[i])**2).sum(axis = 1)
    return np.argmin(c, axis = 1)

def computeCentroids(X, centroids):
    K = len(centroids)
    Ck = findClosestCentroids(X, centroids)
    for i in range(K):
        d = (Ck == i).sum() + 1
        centroids[i] = X[np.where(Ck == i)].sum(axis = 0)/d
    return centroids

def kMeansInitCentroids(X, K):
    m = len(X)
    centroids = np.random.random_integers(0, m, K)
    centroids = X[centroids]
    return centroids

def runKmeans(X, K):
    initCentroids = kMeansInitCentroids(X, K)
    for i in range(50):
        if X.shape[1] == 2:
            plt.scatter(initCentroids[:, 0], initCentroids[:, 1], c = "Black",  marker = 'x')
        initCentroids = computeCentroids(X, initCentroids)
    return initCentroids

def visualiseClusters(X, K):
    finalCentroids = runKmeans(X, K)
    clusters = findClosestCentroids(X, finalCentroids)
    for i in range(K):
        Xc = X[np.where(clusters == i)]
        plt.scatter(Xc[:, 0], Xc[:, 1], c = "Black", marker = '.')
    plt.show()


visualiseClusters(X, K)


fileName = "bird_small.png"
compFileName = "bird_small_compressed.png"
K = 16

image = plt.imread(fileName)
plt.imshow(image)
plt.show()

image = image.reshape((len(image) ** 2, 3))

compImage = np.zeros_like(image)

colors = runKmeans(image, K)
clustersIdx = findClosestCentroids(image, colors)

for i in range(K):
    compImage[np.where(clustersIdx == i)] = colors[i]
compImage = compImage.reshape((128, 128, 3))

plt.imshow(compImage)
plt.imsave(compFileName, compImage)
plt.show()