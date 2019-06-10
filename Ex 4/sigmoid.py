# -*- coding: utf-8 -*-
from numpy import exp

def sigmoid(z):
    return 1/(1 + exp(-z))
