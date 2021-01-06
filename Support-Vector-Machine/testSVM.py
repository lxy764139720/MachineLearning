import numpy as np
from smoSimple import smoSimple
import svmImproved
import originalSVM
import jianshu_SVM


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)


data, label = loadDataSet('./testSet.txt')

b, alpha = svmImproved.svm(data, label.reshape((-1, 1)), 1e9, 1e-6, 100, ('rbf', 10))
print("alpha:")
print(alpha)
w = svmImproved.weight(data, label.reshape((-1, 1)), alpha.reshape((-1, 1)))
print("w:")
print(w)
print("b:")
print(b)

import matplotlib.pyplot as plt

support_xlim = []
support_ylim = []
for i in range(100):
    if alpha[i] > 0.0:
        support_xlim.append(data[i, 0])
        support_ylim.append(data[i, 1])
plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], color='b')
plt.scatter(data[label == -1][:, 0], data[label == -1][:, 1], color='r')
plt.scatter(support_xlim, support_ylim, color='g')
plt.show()
