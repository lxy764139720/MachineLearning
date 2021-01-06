import time

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    # filename是待读取文件的文件名或路径+文件名
    dataMat = [];
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def randPickj(i, m):
    # i是alphai的i值，整数; m是alpha个数; j不能等于i
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


class optStruct:
    def __init__(self, data, label, C, toler):
        # 全局变量
        self.X = data
        self.labelMatrix = label
        self.C = C
        self.toler = toler
        self.m = data.shape[0]  # m为样本数

        # 初始化alpha矩阵、b、Es矩阵
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.Es = np.mat(np.zeros((self.m, 2)))  # 缓存误差，两列，第一列表示当前Ei是否有效，第二列表示当前的Ei值
        self.b = 0


def calcEk(oS, k):
    gxk = float(np.multiply(oS.alphas, oS.labelMatrix).transpose() * (oS.X * oS.X[k, :].transpose())) + oS.b
    Ek = gxk - float(oS.labelMatrix[k])
    return Ek


# 选择相较ai具有最大步长(即Ei-Ej)的aj的函数
def selectJ(oS, i, Ei):
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0  # DeltaE表示Ei-Ej,k表示DeltaE最大的样本点索引值，最终会将Ek赋值给Ej
    oS.Es[i] = [1, Ei]  # 使Es矩阵第i位有效
    validEsList = np.nonzero(oS.Es[:, 0].A)[0]  # 将Es矩阵中有效的Ei对应的索引值选出来，作为挑选j的池子

    if len(validEsList) > 1:
        for k in validEsList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE;
                maxK = k;
                Ej = Ek
        return maxK, Ej
    else:  # 若validEsList只有一个Ei有效(初次循环)，则随机选取一个j
        j = randPickj(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.Es[k] = [1, Ek]


def innerL(i, oS):
    Ei = calcEk(oS, i)

    # 判断Ei是否是违反KKT条件超过toler的点，若是再继续挑选j
    if (oS.labelMatrix[i] * Ei < -oS.toler and oS.alphas[i] < oS.C) or (
            oS.labelMatrix[i] * Ei > oS.toler and oS.alphas[i] > 0):
        j, Ej = selectJ(oS, i, Ei)
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy()

        # 计算L，H
        if oS.labelMatrix[i] != oS.labelMatrix[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])  # 这里alpha[i]仍然等于alphaIold
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0  # 第一个跳出条件(跳出本次内循环，遍历下一个alpha进行更新)

        # 计算eta
        eta = oS.X[i, :] * oS.X[i, :].transpose() + oS.X[j, :] * oS.X[j, :].transpose() - 2.0 * oS.X[i, :] * oS.X[j,
                                                                                                             :].transpose()
        if eta == 0:
            print("eta=0")
            return 0  # 第二个跳出条件(因为eta=0不好处理，且出现情况较少，因此这里咱不处理，直接跳出)

        # 根据统计学习方法中的结果公式得到alphaj的解析解，并更新Ej值
        oS.alphas[j] = oS.alphas[j] + oS.labelMatrix[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 更新Ej值

        # 检验alphaj与alphaJold是否有足够大的改变，若改变不够大，说明与alpha旧值没有什么差异，跳出本次内循环
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0  # 第三个跳出条件

        # 约束条件让我们可以根据alphaJ求出alphaI
        oS.alphas[i] = oS.alphas[i] + oS.labelMatrix[i] * oS.labelMatrix[j] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)  # 更新Ei值

        # 更新b值,根据alpha是否在0～C决定更新的b值
        b1 = -Ei - oS.labelMatrix[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].transpose() \
             - oS.labelMatrix[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[i, :].transpose() + oS.b

        b2 = -Ej - oS.labelMatrix[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].transpose() \
             - oS.labelMatrix[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].transpose() + oS.b

        # 若ai或aj在(0,C)之间，则取b=bi或b=bj，若ai aj都不在(0,C)之间，取均值
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1  # 若执行到这里都没有return0跳出，说明已经完成了一个alpha对的更新，返回一个1

    else:
        return 0  # 若ai不足够违反KKT条件，则return0跳出本次内循环


def SMOpro(data, label, C, toler, maxIter, kTup=("lin", 0)):
    oS = optStruct(np.mat(data), np.mat(label).transpose(), C, toler)
    iter = 0;
    entireSet = True;
    alphaPairsChanged = 0

    # 当迭代次数达到上限(这里的迭代次数只要完成一次循环遍历就+1，不论该次循环遍历是否修改了alpha对)，或全集再无可修改的alpha对时，循环停止，计算完成
    while (iter < maxIter) and (entireSet or alphaPairsChanged > 0):
        alphaPairsChanged = 0
        if entireSet:  # 全集遍历
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullset, iter:%d i:%d, pairsChanged: %d" % (iter, i, alphaPairsChanged))
            iter += 1  # 这里的迭代次数只要完成一次循环遍历就+1，不论该次循环遍历是否修改了alpha对

        else:  # 边界遍历
            boundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]  # 选择0<alpha<C的样本点的索引值(即边界点)
            for i in boundIs:
                alphaPairsChanged += innerL(i, oS)
                print("bound, iter:%d i:%d, pairsChanged: %d" % (iter, i, alphaPairsChanged))
            iter += 1

        # 控制遍历往返于全集遍历和边界遍历
        if entireSet:
            entireSet = False  # 若本轮是全集遍历，则下一轮进入边界遍历(下一轮while条件中的entire是False)
        elif alphaPairsChanged == 0:
            entireSet = True  # 若本轮是边界遍历，且本轮遍历未修改任何alpha对，则下一轮进入全集遍历
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def weight(data, label, alphas):
    dataMatrix = np.mat(data);
    labelMatrix = np.mat(label).transpose()  # 这里labelMatrix形状为m*1
    m, n = dataMatrix.shape
    w = np.mat(np.zeros((1, n)))  # 初始化w，为1行n列的全零矩阵，n为data维度数

    """w=求和(ai*yi*xi),求和对象是支持向量，即，ai>0的样本点，xi，yi为支持向量对应的label和data"""
    for i in range(m):
        if alphas[i] > 0:
            w += labelMatrix[i] * alphas[i] * dataMatrix[i, :]
    return w.tolist()


"""
绘制样本数据以及决策边界
思路：
1. 将样本数据根据样本类别标签labelMat分别放入不同的坐标集中
2. 根据坐标集合，分别绘制两个类别样本的散点图
3. 决策边界即x2=f(x1),由w1*x1+w2*x2+b=0得到x2(即y=(-b-w1x1)/w2)
"""


def plotBestFit(weights, b, filename, alpha):
    dataMat, labelMat = loadDataSet(filename)  # 加载样本特征、样本类别
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]  # n个样本
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []  # 两个类别的样本的xy坐标值，x对应x1,y对应x2
    x_support_cord = []
    y_support_cord = []  # 支持向量

    # 将样本数据根据样本类别标签labelMat分别放入不同的坐标集中
    for i in range(n):
        if int(labelMat[i]) == 1:  # 第i个样本是1类
            xcord1.append(dataArr[i, 0])  # 第i个样本的x1值
            ycord1.append(dataArr[i, 1])  # 第i个样本的x2值
        else:
            xcord2.append(dataArr[i, 0])  # 第i个样本的x1值
            ycord2.append(dataArr[i, 1])  # 第i个样本的x2值
        if alpha[i] > 0:
            x_support_cord.append(dataArr[i, 0])
            y_support_cord.append(dataArr[i, 1])

    # 绘制两类样本的散点图
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(xcord1, ycord1, c="red", s=50, label="label=1")
    plt.scatter(xcord2, ycord2, c="blue", s=50, label="label=-1")  # 继续在原图上作图
    plt.scatter(x_support_cord, y_support_cord, c="green", s=50, label="support")  # 继续在原图上作图

    # 绘制决策边界
    x = np.arange(-3.0, 5.0, 0.1)
    y = (-b - weights[0][0] * x) / weights[0][1]  # 由w1*x1+w2*x2+b=0得到x2(即y)=(-b-w1x1)/w2
    x.shape = (len(x), 1)
    y.shape = (len(x), 1)
    plt.plot(x, y, color="darkorange", linewidth=3.0, label="Boarder")  # 继续在ax图上作图

    plt.xlabel("X1", fontsize=16)
    plt.ylabel("X2", fontsize=16)
    plt.title("SMO BestFit", fontsize=20, fontweight="bold")
    plt.legend()  # 添加图标注解
    plt.show()


if __name__ == '__main__':
    data3, label3 = loadDataSet("testSet.txt")
    start = time.time()
    b3, alphas3 = SMOpro(data3, label3, 0.6, 0.001, 60)
    print("\n", "time used:.{0}s".format(time.time() - start))
    w3 = weight(data3, label3, alphas3)
    plotBestFit(w3, b3, "testSet.txt", alphas3)
