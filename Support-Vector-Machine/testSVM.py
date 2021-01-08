import numpy as np
from smoSimple import smoSimple
import svmImproved
import originalSVM
import jianshu_SVM
from svm import SVC


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

svm = SVC(kernel='linear', probability=True, verbose=True)
result = svm.fit(data, label.reshape((-1, 1)))
print("alpha:")
print(result.alpha_)
# w = svmImproved.weight(data, label.reshape((-1, 1)), result.alpha_.reshape((-1, 1)))
print("w:")
print(result.w_)
print("b:")
print(result.b_)
print("support_vectors:")
print(result.support_vectors_)
svm.predict(np.array([[5, 2]]))
print("prob:")
print(result.predict_prob_)

import matplotlib.pyplot as plt

support_xlim = []
support_ylim = []
for i in range(100):
    if result.alpha_[i] > 0.0:
        support_xlim.append(data[i, 0])
        support_ylim.append(data[i, 1])
plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], color='b')
plt.scatter(data[label == -1][:, 0], data[label == -1][:, 1], color='r')
plt.scatter(support_xlim, support_ylim, color='g')

x = np.arange(-0.0, 8.0, 0.1)
y = (-result.b_ - result.w_[0] * x) / result.w_[1]  # 由w1*x1+w2*x2+b=0得到x2(即y)=(-b-w1x1)/w2
x.shape = (len(x), 1)
y.shape = (len(x), 1)
plt.plot(x, y, color="darkorange", linewidth=3.0, label="Boarder")  # 继续在ax图上作图

plt.show()
