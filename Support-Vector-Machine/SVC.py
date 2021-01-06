import numpy as np
from sklearn.metrics import accuracy_score


class SVC:
    def __init__(self, C=1e9, kernel='rbf', gamma='scale', coef0=0, degree=3, epsilon=1e-3, max_steps=np.inf,
                 verbose=False):
        assert C > 0, "C must be greater than 0"
        assert epsilon > 0, "epsilon must be greater than 0"
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.verbose = verbose
        # 私有变量
        self._X = None
        self._Y = None
        # 计算值
        self.alpha_ = None
        self.b_ = None
        self.w_ = None

    class OptStruct:
        def __init__(self, X, Y, C, epsilon, kernel, gamma, coef0, degree):
            assert X.shape[0] == Y.shape[0], "the size of X must be equal to the size of y"
            assert C > 0, "C must be greater than 0"
            assert epsilon > 0, "epsilon must be greater than 0"
            assert kernel == 'rbf' and gamma > 0, "gamma of rbf kernel must be greater than 0"
            self.X = X
            self.Y = Y
            self.C = C
            self.epsilon = epsilon
            self.m = np.shape(X)[0]
            self.alpha = np.zeros((self.m, 1))
            self.b = 0
            # ECache[0]:eCache是否有效（已计算好），ECache[1]:实际的E值
            self.ECache = np.zeros((self.m, 2))
            self.K = np.zeros((self.m, self.m))
            for i in range(self.m):
                self.K[:, i] = SVC.kernelTrans(self.X, self.X[i, :], kernel, gamma, coef0, degree)

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], "the size of X must be equal to the size of y"
        if self.kernel == 'rbf':
            if self.gamma == 'scale':
                # gamma='scale' 时，gamma=1/(n_features*X.var())
                self.gamma = 1 / (X.shape[1] * np.var(X))
            elif self.gamma == 'auto':
                # gamma='auto' 时，gamma=1/n_features
                self.gamma = 1 / X.shape[1]
        opt = self.OptStruct(np.array(X), np.array(Y), self.C, self.epsilon, self.kernel, self.gamma, self.coef0,
                             self.degree)
        iteration = 0
        entireSet = True  # 冷热数据分离，热数据：0<alpha<C，冷数据：alpha<=0 | alpha>=C
        alphaPairsChanged = 0
        # 第一层循环：循环少于最大次数 且 上一次alpha改变过或在部分数据集上alpha未改变
        while iteration < self.max_steps and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            # 遍历全部数据集，对每个alpha进行优化
            if entireSet:
                for i in range(opt.m):
                    alphaPairsChanged += self._smo(i, opt)
                    if self.verbose:
                        print("全部遍历, iter:{} i:{}, pairs changed {}".format(iteration, i, alphaPairsChanged))
                iteration += 1
            # 遍历热数据
            else:
                # 非边界值：0<alpha<C
                nonBound = np.nonzero((opt.alpha > opt.epsilon) * (opt.alpha < self.C - opt.epsilon))[0]
                for i in nonBound:
                    alphaPairsChanged += self._smo(i, opt)
                    if self.verbose:
                        print("非边界值遍历, iter:{} i:{}, pairs changed {}".format(iteration, i, alphaPairsChanged))
                iteration += 1
            # 如果该次遍历了全部的alpha，则下次遍历热数据中的alpha
            if entireSet:
                entireSet = False
            # 如果热数据中的alpha没有可更新的，就遍历更新全部的alpha
            elif alphaPairsChanged == 0:
                entireSet = True
            if self.verbose:
                print("iteration number: %d".format(iteration))
        self.alpha_ = opt.alpha
        self.b_ = opt.b
        self.w_ = self._weight(X, Y, self.alpha_)
        return self

    @classmethod
    def kernelTrans(cls, X, A, kernel, gamma, coef0, degree):
        m, n = np.shape(X)
        K = np.zeros((m, 1))
        if kernel == 'linear':
            K = X.dot(A.T)
        elif kernel == 'poly':
            K = np.power(gamma * X.dot(A.T) + coef0, degree)
        elif kernel == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow.dot(deltaRow.T)
            K = np.exp(K / (-1 * gamma ** 2))
        elif kernel == 'sigmoid':
            K = np.tanh(gamma * X.dot(A.T) + coef0)
        else:
            raise NameError('The kernel name is not recognized')
        return K.flatten()

    def _calcEk(self, opt, k):
        """
        计算alpha[k]的误差Ek
        g(xi)=sum(alpha_i*yi*K(xi,x))+b
        Ei=g(xi)-yi
        """
        g_xk = (np.multiply(opt.alpha, opt.Y).T.dot(opt.K[:, k]) + opt.b).astype(np.float)
        Ek = g_xk - float(opt.Y[k])
        return Ek

    def _selectJ(self, i, opt, Ei):
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
                Ek = self._calcEk(opt, k)
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
            Ej = self._calcEk(opt, j)
        return j, Ej

    def _updateEk(self, opt, k):
        # 更新缓存
        Ek = self._calcEk(opt, k)
        opt.ECache[k] = [1, Ek]

    def _smo(self, i, opt):
        """
        内层循环，给定要优化的alpha[i]，找到最优的alpha[j]，并对这一对(i,j)进行优化
        :return: 1：i,j被更新，0：i,j未被更新
        """
        Ei = self._calcEk(opt, i)
        # 第一个alpha变量的选择：选择不满足KKT条件的样本点(X[i],Y[i])和alpha[i]，满足以下任意一条即满足KKT条件
        # alpha[i]==0 且 Y[i]*g(xi)>=1（即Y[i]*Ei>=0.0）
        # 0<alpha[i]<C 且 Y[i]*g(xi)==1（即Y[i]*Ei==0.0）
        # alpha[i]==C 且 Y[i]*g(xi)<=1（即Y[i]*Ei<=0.0）
        if ((np.abs(opt.Y[i] * Ei) > opt.epsilon) and (0 < opt.alpha[i] < opt.C)) or \
                ((opt.Y[i] * Ei < opt.epsilon) and (opt.alpha[i] == 0)) or \
                ((opt.Y[i] * Ei > opt.epsilon) and (opt.alpha[i] == opt.C)):

            # 第二个alpha变量的选择：使alpha[j]的变化最大
            j, Ej = self._selectJ(i, opt, Ei)
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
                if self.verbose:
                    print("L==H")
                return 0

            # 更新alpha_j
            # eta=K11+K22-2K12
            eta = opt.K[i, i] + opt.K[j, j] - 2.0 * opt.K[i, j]
            if eta <= 0:
                if self.verbose:
                    print("eta<=0")
                return 0
            opt.alpha[j] += opt.Y[j] * (Ei - Ej) / eta
            opt.alpha[j] = min(opt.alpha[j], H)  # 剪辑上界
            opt.alpha[j] = max(opt.alpha[j], L)  # 剪辑下界
            self._updateEk(opt, j)  # 更新缓存
            if abs(opt.alpha[j] - alphaJold) < 0.00001:
                if self.verbose:
                    print("j not moving enough")
                return 0

            # 更新alpha_i
            opt.alpha[i] += opt.Y[j] * opt.Y[i] * (alphaJold - opt.alpha[j])
            self._updateEk(opt, i)

            # 更新b
            b1 = opt.b - Ei - opt.Y[i] * (opt.alpha[i] - alphaIold) * opt.K[i, i] - opt.Y[j] * (
                    opt.alpha[j] - alphaJold) * opt.K[i, j]
            b2 = opt.b - Ej - opt.Y[i] * (opt.alpha[i] - alphaIold) * opt.K[i, j] - opt.Y[j] * (
                    opt.alpha[j] - alphaJold) * opt.K[j, j]
            if 0 < opt.alpha[i] < opt.C:
                opt.b = b1
            elif 0 < opt.alpha[j] < opt.C:
                opt.b = b2
            else:
                opt.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def _weight(self, X, Y, alpha):
        w = (Y * alpha * X).sum(axis=0)
        return w

    def predict(self, X_test):
        assert self.w_ is not None, "must fit before predict"
        assert X_test.shape[1] == len(self.w_), "the feature number of X_predict must be equal to X_train"
        # Y_predict = sign(sum(alpha*Y*K(X,X_test))+b)
        K = np.zeros((self._X.shape[0], X_test.shape[0]))
        for i in range(X_test):
            K[:, i] = self.kernelTrans(self._X, X_test[i, :], self.kernel, self.gamma, self.coef0, self.degree)
        score = (self.alpha_ * self._Y * K).sum(axis=0) + self.b_
        Y_predict = (score >= 0).astype(int) * 2 - 1
        return Y_predict

    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return accuracy_score(Y_test, Y_predict)

    def __repr__(self):
        return "SVC(C={}, gamma={})".format(self.C, self.gamma)
