# -*- coding: utf-8 -*-
import re
import numpy as np
import nltk.stem.porter as pt
from scipy.io import loadmat
from sklearn.svm import SVC

def emailProcess(emailSample):   
    samp1 = open(emailSample, "r").read()
    samp1 = samp1.lower()
    samp1 = samp1.replace("\n", " ")
    samp1 = re.sub('[0-9]+', 'number', samp1)
    samp1 = re.sub('[$]+', 'dollar', samp1)
    samp1 = re.sub('[^ ]+@[^ ]+', 'emailaddr', samp1)
    samp1 = re.sub('(http|https)://[^ ]+', 'httpaddr', samp1)
    samp1 = re.sub('<[^ ]+>', '', samp1)
    samp1 = re.sub('[^a-zA-Z\s]', '', samp1)
    samp1 = " ".join(samp1.split())
    tokens = samp1.split()
    stemmer = pt.PorterStemmer()
    for i in range(len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])
    return tokens

def featureExtraction(vocab_values, tokens):
    indices = [i for i in range(len(vocab_values)) if vocab_values[i] in tokens]
    f_vec = np.zeros((1, len(vocab_values)))
    for i in indices:
        f_vec[0, i] = 1
    return f_vec

vocab_dict = {}
with open("vocab.txt") as f:
    for line in f:
        key, val = line.split()
        vocab_dict[int(key)] = val
vocab_values = list(vocab_dict.values())

spamTrain = loadmat("spamTrain.mat")
X_Train = spamTrain['X']
y_Train = spamTrain['y']
m1 = X_Train.shape[0]

spamTest = loadmat("spamTest.mat")
X_Test = spamTest['Xtest']
y_Test = spamTest['ytest']
m2 = X_Test.shape[0]

clf = SVC(C = 0.1, kernel = 'linear')
clf.fit(X_Train, y_Train.flatten())

accuracyTrain = (clf.predict(X_Train) == y_Train.flatten()).sum()/m1
accuracyTest = (clf.predict(X_Test) == y_Test.flatten()).sum()/m2

print("Training accuracy: ", accuracyTrain *100)
print("Test accuracy: ", accuracyTest * 100)

samp1 = featureExtraction(vocab_values, emailProcess("emailSample1.txt"))
samp2 = featureExtraction(vocab_values, emailProcess("emailSample2.txt"))
spam1 = featureExtraction(vocab_values, emailProcess("spamSample1.txt"))
spam2 = featureExtraction(vocab_values, emailProcess("spamSample2.txt"))

predicts = list(map(clf.predict, [samp1, samp2, spam1, spam2]))

print("Classifier predictions {1 - Spam, 0 - Not Spam} for emailSample1, emailSample2, spamSample1 and spamSample2: ", predicts[0][0], predicts[1][0], predicts[2][0], predicts[3][0] )

coeff = np.argsort(clf.coef_).flatten()[::-1]
print("The top predictors for spam are:")
for i in range(15):
    print(vocab_values[coeff[i]])
