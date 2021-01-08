import numpy as np
from svm import SVC
# from sklearn.svm import SVC
# from otherSVM import SVM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat), np.array(labelMat)


# 绘制决策边界
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new).reshape(x0.shape)
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D'])
    plt.contourf(x0, x1, y_predict, cmap=custom_cmap)


def gaussian_kernel(x, y):
    return np.exp(-((x - y) ** 2).sum())


if __name__ == '__main__':
    # data, label = loadDataSet('testSetRBF2.txt')
    dataSet = np.loadtxt('./round_scatter_simple.csv', delimiter=",")
    data = dataSet[:, :2]
    label = dataSet[:, -1]
    svm = SVC(C=1e9, kernel='rbf', gamma='scale', probability=True, decision_function_shape='ovo')
    result = svm.fit(data, label.reshape((-1, 1)))

    support_vectors = svm.support_vectors_
    # support_xlim = []
    # support_ylim = []
    # for i in range(len(support_vectors)):
    #     support_xlim.append(support_vectors[i, 0])
    #     support_ylim.append(support_vectors[i, 1])

    plot_decision_boundary(svm, axis=[0, 2.7, 0, 2.7])
    plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], color='b')
    plt.scatter(data[label == 2][:, 0], data[label == 2][:, 1], color='r')
    # plt.scatter(support_xlim, support_ylim, color='g')

    plt.show()
