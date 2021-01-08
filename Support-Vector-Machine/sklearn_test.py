import numpy as np
# from sklearn.svm import SVC
from svm import SVC
from sklearn.datasets import load_wine

# def loadDataSet(fileName):
#     dataMat = []
#     labelMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         lineArr = line.strip().split('\t')
#         dataMat.append([float(lineArr[0]), float(lineArr[1])])
#         labelMat.append(float(lineArr[2]))
#     return np.array(dataMat), np.array(labelMat)
#
#
# data, label = loadDataSet('./testSet.txt')

dataset = load_wine()
data = dataset['data']
label = dataset['target']

svm = SVC(C=1, kernel='rbf', probability=True)
# b, alpha = smoSimple(data, label, 1e9, 1e-6, 100)
svm.fit(data, label.reshape((-1, 1)))
# print(b)
# print(alpha)
# w = ((alpha * label)[:, None] * data).sum(axis=0)
# print(w)

s = svm.support_vectors_
print(s)
support_xlim = s[:, 0]
support_ylim = s[:, 1]

import matplotlib.pyplot as plt

# support_xlim = []
# support_ylim = []
# for i in range(100):
#     if alpha[i] > 0.0:
#         support_xlim.append(data[i, 0])
#         support_ylim.append(data[i, 1])
plt.scatter(data[label == 0][:, 0], data[label == 0][:, 1], color='b')
plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], color='g')
plt.scatter(data[label == 2][:, 0], data[label == 2][:, 1], color='r')
plt.scatter(support_xlim, support_ylim, color='y')
plt.show()
