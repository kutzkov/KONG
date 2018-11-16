#!/usr/bin/env python

"""
The usual train-test split mumbo-jumbo

Please contact Konstantin Kutzkov (kutzkov@gmail.com) if you have any questions.
"""

# The usual train-test split mumbo-jumbo
#from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
import platform
import numpy as np
import time


def normalize_vector(v):
    norm2 = np.linalg.norm(np.array(v))
    if norm2 == 0:
        norm2 = 1
    return [x/norm2 for x in v]

def get_accuracy(y_true, y_pred):
    cnt = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            cnt += 1
    return cnt/len(y_true)

def read_data(filename, classval):
    f = open(filename, 'r')
    cnt = 0
    X = []
    y = []
    for line in f:
        if cnt % 2 == 0:
            #print(len(line.split()))
            X.append(line.split())
        else:
            c = int(line.strip())
            if c != classval:
                c = classval + 1
            y.append(c)
        cnt += 1
    X = np.reshape(X, (cnt//2, -1))
    return X, np.array(y)

def read_graphs(filename):
    f = open(filename, 'r')
    cnt = 0
    X = []
    y = []
    for line in f:
        if cnt % 2 == 0:
            #print(len(line.split()))
            X.append(line.split())
        else:
            y.append(int(line.strip()))
#            print(line)
#            print(int(line.strip()))
#            print('\n')
        cnt += 1
    X = np.reshape(X, (cnt//2, -1))
    return X, np.array(y)

def normalize_data(X):
    for i in range(X.shape[0]):
        X[i, :] = normalize_vector(X[i, :])
        