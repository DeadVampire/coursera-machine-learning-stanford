# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np


def cost(X, Theta, ratings, given):
    l = 1.5
    J = (((X@Theta.T - ratings)**2) * given).sum()/2
    J += ((Theta**2).sum() + (X**2).sum()) * (l/2)
    return J

def grad(X, Theta, ratings, given):
    l = 1.5
    temp = (X@Theta.T - ratings) * given
    gradX = temp@Theta + (l * X)
    gradTheta = temp.T@X + (l * Theta)
    return gradX, gradTheta


data = loadmat("ex8_movies.mat")

ratings = data['Y']
given = data['R']

movies = []
with open("movie_ids.txt", encoding = 'latin') as f:
    for line in f:
        movies.append(' '.join(line.split()[1:]))

myRatings = np.zeros((len(ratings), 1))
myRatings[0] = 4
myRatings[6] = 3
myRatings[11]= 5
myRatings[53]= 4
myRatings[63]= 5
myRatings[65]= 3
myRatings[68]= 5
myRatings[97] = 2
myRatings[182]= 4
myRatings[225]= 5
myRatings[354]= 5

ratings = np.hstack((ratings, myRatings))
given = np.hstack((given, myRatings != 0 ))

mean = (ratings.sum(axis = 1)/given.sum(axis = 1)).reshape((-1, 1))
ratings = ratings - mean

X = np.random.normal(size = (1682, 10))
Theta = np.random.normal(size = (944, 10))

alpha = 0.005
for i in range(400):
    grads = grad(X, Theta, ratings, given)
    Theta = Theta - alpha * grads[1] 
    X = X - alpha * grads[0]

predRating = X@Theta.T + mean
idx = np.argsort(predRating[:, -1])

print("Top recommended movies for you: ")
for i in range(1, 11):
    print(movies[idx[-i]])
