import numpy as np
from smoSimple import smoSimple
import matplotlib.pyplot as plt


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
b, alphas = smoSimple(data, label, C=0.6, toler=0.001, maxIter=40)
w = ((alphas * label)[:, None] * data).sum(axis=0)
print(w)

support_xlim = []
support_ylim = []
for i in range(100):
    if alphas[i] > 0.0:
        support_xlim.append(data[i, 0])
        support_ylim.append(data[i, 1])
plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], color='b')
plt.scatter(data[label == -1][:, 0], data[label == -1][:, 1], color='r')
plt.scatter(support_xlim, support_ylim, color='g')

plt.show()
