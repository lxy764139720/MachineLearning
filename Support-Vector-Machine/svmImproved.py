import numpy as np


class OptStruct:
    def __init__(self, X, Y, C, epsilon):
        self.X = X
        self.Y = Y
        self.C = C
        self.epsilon = epsilon
        self.m = np.shape(X)[0]
        self.alpha = np.zeros((self.m, 1))
        self.b = 0
        # ECache[0]:eCache是否有效（已计算好），ECache[1]:实际的E值
        self.ECache = np.zeros((self.m, 2))


def calcEk(opt, k):
    """
    计算alpha[k]的误差Ek
    g(xi)=sum(alpha_i*yi*K(xi,x))+b
    Ei=g(xi)-yi
    """
    g_xk = float(np.multiply(opt.alpha, opt.Y).T * (opt.X * opt.Y[k, :].T)) + opt.b
    Ek = g_xk - float(opt.Y[k])
    return Ek


def selectJ(i, opt, Ei):
    """
    选择alpha中使误差Ej和Ei相差最大的j
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    validECacheList = np.nonzero(opt.ECache[:, 0])[0]
    # 如果ECache已有更新，选择Ei和Ej相差最大的j
    if len(validECacheList) > 1:
        for k in validECacheList:
            if k == i:
                continue
            Ek = calcEk(opt, k)
            deltaE = np.abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    # 如果ECache未改变过（第一次选择），随机选择j
    else:
        j = i
        while j == i:
            j = int(np.random.uniform(0, opt.m))
        Ej = calcEk(opt, j)
    return j, Ej


def updateEk(opt, k):
    # 更新缓存
    Ek = calcEk(opt, k)
    opt.ECache[k] = [1, Ek]


def smo(i, opt):
    """
    内层循环，给定要优化的alpha[i]，找到最优的alpha[j]，并对这一对(i,j)进行优化
    :return: 1：i,j被更新，0：i,j未被更新
    """
    Ei = calcEk(opt, i)
    # 第一个alpha变量的选择：选择不满足KKT条件的样本点(X[i],Y[i])和alpha[i]，满足以下任意一条即满足KKT条件
    # alpha[i]==0 且 Y[i]*g(xi)>=1（即Y[i]*Ei>=0.0）
    # 0<alpha[i]<C 且 Y[i]*g(xi)==1（即Y[i]*Ei==0.0）
    # alpha[i]==C 且 Y[i]*g(xi)<=1（即Y[i]*Ei<=0.0）
    if ((np.abs(opt.Y[i] * Ei) > opt.epsilon) and (0 < opt.alpha[i] < opt.C)) or \
            ((opt.Y[i] * Ei < opt.epsilon) and (opt.alpha[i] == 0)) or \
            ((opt.Y[i] * Ei > opt.epsilon) and (opt.alpha[i] == opt.C)):

        # 第二个alpha变量的选择：使alpha[j]的变化最大
        j, Ej = selectJ(i, opt, Ei)
        alphaIold = opt.alpha[i].copy()
        alphaJold = opt.alpha[j].copy()

        # 计算上下界
        if opt.Y[i] != opt.Y[j]:
            L = max(0, opt.alpha[j] - opt.alpha[i])
            H = min(opt.C, opt.C + opt.alpha[j] - opt.alpha[i])
        else:
            L = max(0, opt.alpha[j] + opt.alpha[i] - opt.C)
            H = min(opt.C, opt.alpha[j] + opt.alpha[i])
        if L == H:
            print("L==H")
            return 0

        # 更新alpha_j
        # eta=K11+K22-2K12
        eta = opt.X[i, :] * opt.X[i, :].T + opt.X[j, :] * opt.X[j, :]. \
            T - 2.0 * opt.X[i, :] * opt.X[j, :].T
        if eta <= 0:
            print("eta<=0")
            return 0
        opt.alpha[j] += opt.Y[j] * (Ei - Ej) / eta
        opt.alpha[j] = min(opt.alpha[j], H)  # 剪辑上界
        opt.alpha[j] = max(opt.alpha[j], L)  # 剪辑下界
        updateEk(opt, j)  # 更新缓存
        if abs(opt.alpha[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0

        # 更新alpha_i
        opt.alpha[i] += opt.Y[j] * opt.Y[i] * (alphaJold - opt.alpha[j])
        updateEk(opt, i)

        # 更新b
        b1 = opt.b - Ei - opt.Y[i] * (opt.alpha[i] - alphaIold) * opt.X[i, :] * opt.X[i, :]. \
            T - opt.Y[j] * (opt.alpha[j] - alphaJold) * opt.X[i, :] * opt.X[j, :].T
        b2 = opt.b - Ej - opt.Y[i] * (opt.alpha[i] - alphaIold) * opt.X[i, :] * opt.X[j, :]. \
            T - opt.Y[j] * (opt.alpha[j] - alphaJold) * opt.X[j, :] * opt.X[j, :].T
        if 0 < opt.alpha[i] < opt.C:
            opt.b = b1
        elif 0 < opt.alpha[j] < opt.C:
            opt.b = b2
        else:
            opt.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def svm(X, Y, C, tolerance, maxIter, kTup=('lin', 0)):
    opt = OptStruct(np.array(X), np.array(Y).T, C, tolerance)
    iteration = 0
    entireSet = True  # 冷热数据分离，热数据：0<alpha<C，冷数据：alpha<=0 | alpha>=C
    alphaPairsChanged = 0
    # 第一层循环：循环少于最大次数 且 上一次alpha改变过或在部分数据集上alpha未改变
    while iteration < maxIter and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        # 遍历全部数据集，对每个alpha进行优化
        if entireSet:
            for i in range(opt.m):
                alphaPairsChanged += smo(i, opt)
                print("全部遍历, iter:%d i:%d, pairs changed %d".format(iteration, i, alphaPairsChanged))
                iteration += 1
        # 遍历热数据
        else:
            nonBoundIs = np.nonzero((opt.alpha > 0) * (opt.alpha < C))[0]  # 非边界值：0<alpha<C
            for i in nonBoundIs:
                alphaPairsChanged += smo(i, opt)
                print("非边界值遍历, iter: %d i:%d, pairs changed %d".format(iteration, i, alphaPairsChanged))
                iteration += 1
        # 如果该次遍历了全部的alpha，则下次遍历热数据中的alpha
        if entireSet:
            entireSet = False
        # 如果热数据中的alpha没有可更新的，就遍历更新全部的alpha
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d".format(iteration))
    return opt.b, opt.alpha
