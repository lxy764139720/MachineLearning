from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score


class SVC:
    def __init__(self, C=1e9, kernel='rbf', gamma='scale', coef0=0, degree=3, epsilon=1e-3, max_steps=np.inf,
                 decision_function_shape='ovr', probability=False, verbose=False):
        assert C > 0, "C must be greater than 0"
        assert epsilon > 0, "epsilon must be greater than 0"
        assert decision_function_shape == 'ovo' or decision_function_shape == 'ovr', \
            "decision_function_shape must be ovo or ovr"
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.decision_function_shape = decision_function_shape
        self.probability = probability
        self.verbose = verbose
        # 私有变量
        self._X = None
        self._Y = None  # 规定二分类正例为1，负例为-1
        # 计算值
        self.alpha_ = None
        self.b_ = None
        self.w_ = None
        self.classes_ = None
        self.alpha_num_ = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.K_ = None
        self.score_ = None
        self.prob_A_ = None
        self.prob_B_ = None
        self.predict_prob_ = None  # 预测输出的概率

    class OptStruct:
        def __init__(self, X, Y, C, epsilon, kernel, gamma, coef0, degree):
            assert X.shape[0] == Y.shape[0], "the size of X must be equal to the size of y"
            assert C > 0, "C must be greater than 0"
            assert epsilon > 0, "epsilon must be greater than 0"
            if kernel == 'rbf' and gamma <= 0:
                raise ValueError("gamma of rbf kernel must be greater than 0")
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
        # 计算gamma
        if self.kernel == 'rbf':
            if self.gamma == 'scale':
                # gamma='scale' 时，gamma=1/(n_features*X.var())
                self.gamma = 1 / (X.shape[1] * np.var(X))
            elif self.gamma == 'auto':
                # gamma='auto' 时，gamma=1/n_features
                self.gamma = 1 / X.shape[1]
        opt = self.OptStruct(np.array(X), np.array(Y), self.C, self.epsilon, self.kernel, self.gamma, self.coef0,
                             self.degree)

        # 存储数据
        self._X = X
        self._Y = Y

        counter = Counter(self._Y.flatten())
        self.classes_ = np.array(list(counter.keys()))
        num_classes = len(self.classes_)
        if num_classes == 1:
            raise ValueError("The number of labels must be greater than 1")
        # 处理二分类
        elif num_classes == 2:
            result_dict = self._solve_binary(counter, opt)
            self.alpha_ = result_dict['alpha']
            self.b_ = result_dict['b']
            self.w_ = result_dict['w']
            self.support_vectors_ = result_dict['support_vectors']
            if self.probability:
                self.K_ = result_dict['K']
                self.prob_A_, self.prob_B_ = result_dict['prob_A'], result_dict['prob_B']
        # 处理多分类
        else:
            if self.decision_function_shape == 'ovr':
                self.alpha_num_ = np.array(list(counter.values()))
                self._solve_ovr(opt)
            elif self.decision_function_shape == 'ovo':
                self.alpha_num_ = np.array(list(counter.values()))  # TODO
                self._solve_ovo(opt)
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

    def _inner(self, i, opt):
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

    def _outer(self, opt):
        iteration = 0
        entireSet = True  # 冷热数据分离，热数据：0<alpha<C，冷数据：alpha<=0 | alpha>=C
        alphaPairsChanged = 0
        # 第一层循环：循环少于最大次数 且 上一次alpha改变过或在部分数据集上alpha未改变
        while iteration < self.max_steps and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            # 遍历全部数据集，对每个alpha进行优化
            if entireSet:
                for i in range(opt.m):
                    alphaPairsChanged += self._inner(i, opt)
                    if self.verbose:
                        print("全部遍历, iter:{} i:{}, pairs changed {}".format(iteration, i, alphaPairsChanged))
                iteration += 1
            # 遍历热数据
            else:
                # 非边界值：0<alpha<C
                nonBound = np.nonzero((opt.alpha > opt.epsilon) * (opt.alpha < self.C - opt.epsilon))[0]
                for i in nonBound:
                    alphaPairsChanged += self._inner(i, opt)
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
                print("iteration number: {}".format(iteration))

        result_dict = {'alpha': opt.alpha, 'b': opt.b, 'w': self._weight(opt.X, opt.Y, opt.alpha),
                       'support_vectors': opt.X[opt.alpha.flatten() > 0]}
        # 计算svm sigmoid概率函数的系数
        if self.probability:
            result_dict['K'] = opt.K
            result_dict['prob_A'], result_dict['prob_B'] = self._calc_prob((opt.alpha * opt.Y * opt.K).sum(axis=0),
                                                                           opt.Y)
        return result_dict

    def _solve_binary(self, counter, opt):
        # 处理二分类
        if 1 in counter and -1 in counter:
            result_dict = self._outer(opt)
        elif 1 in counter:
            opt.Y[opt.Y != 1] = -1
            result_dict = self._outer(opt)
        elif -1 in counter:
            opt.Y[opt.Y != -1] = 1
            result_dict = self._outer(opt)
        else:
            opt.Y[opt.Y == opt.Y[0]] = 1
            opt.Y[opt.Y != opt.Y[0]] = -1
            result_dict = self._outer(opt)
        return result_dict

    def _solve_ovr(self, opt):
        # ovr处理多分类
        num_classes = len(self.classes_)
        # 二维数组存储结果
        self.b_ = np.zeros(num_classes)
        self.w_ = np.zeros((num_classes, self._X.shape[1]))
        self.alpha_num_ = np.zeros(num_classes)
        self.n_support_ = np.zeros(num_classes)
        if self.probability:
            self.prob_A_ = np.zeros(num_classes)
            self.prob_B_ = np.zeros(num_classes)

        k = 0
        for i in range(num_classes):
            # 生成只包含两类{1, -1}的X, Y
            i_index = opt.Y == self.classes_[i]
            not_i_index = opt.Y != self.classes_[i]
            opt.Y[i_index] = 1
            opt.Y[not_i_index] = -1
            opt.m = opt.X.shape[0]
            opt.alpha = np.zeros((opt.m, 1))
            opt.K = np.zeros((opt.m, opt.m))
            for a in range(opt.m):
                opt.K[:, a] = SVC.kernelTrans(opt.X, opt.X[a, :], self.kernel, self.gamma, self.coef0, self.degree)
            counter = Counter(opt.Y.flatten())
            result_dict = self._solve_binary(counter, opt)
            if self.alpha_ is None:
                self.alpha_ = result_dict['alpha']
                self.support_vectors_ = result_dict['support_vectors']
                if self.probability:
                    self.K_ = result_dict['K']
            else:
                self.alpha_ = np.vstack((self.alpha_, result_dict['alpha']))
                self.support_vectors_ = np.vstack((self.support_vectors_, result_dict['support_vectors']))
                if self.probability:
                    self.K_ = np.vstack((self.K_, result_dict['K']))
            self.b_[k] = result_dict['b']
            self.w_[k] = result_dict['w']
            self.alpha_num_[k] = len(result_dict['alpha'])
            self.n_support_[k] = len(result_dict['support_vectors'])
            if self.probability:
                self.prob_A_[k] = result_dict['prob_A']
                self.prob_B_[k] = result_dict['prob_B']
            k += 1

        self.alpha_ = np.array(self.alpha_)
        self.support_vectors_ = np.array(self.support_vectors_)
        if self.probability:
            self.K_ = np.array(self.K_)

    def _solve_ovo(self, opt):
        # ovo处理多分类
        num_classes = len(self.classes_)
        # 二维数组存储结果
        self.b_ = np.zeros(int(num_classes * (num_classes - 1) / 2))
        self.w_ = np.zeros((int(num_classes * (num_classes - 1) / 2), self._X.shape[1]))
        self.alpha_num_ = np.zeros(int(num_classes * (num_classes - 1) / 2))
        self.n_support_ = np.zeros(num_classes)
        if self.probability:
            self.prob_A_ = np.zeros(int(num_classes * (num_classes - 1) / 2))
            self.prob_B_ = np.zeros(int(num_classes * (num_classes - 1) / 2))

        k = 0
        for i in range(num_classes):
            for j in range(num_classes):
                # 只计算上三角部分
                if i < j:
                    # 生成只包含两类{1, -1}的X, Y
                    mask = self._Y != self.classes_[i]
                    mask[(self._Y != self.classes_[j]) == False] = False
                    opt.X = np.delete(self._X, mask.flatten(), axis=0)
                    opt.Y = np.delete(self._Y, mask.flatten(), axis=0)
                    i_index = opt.Y == self.classes_[i]
                    j_index = opt.Y == self.classes_[j]
                    opt.m = opt.X.shape[0]
                    opt.Y[i_index] = 1
                    opt.Y[j_index] = -1
                    opt.alpha = np.zeros((opt.m, 1))
                    opt.K = np.zeros((opt.m, opt.m))
                    for a in range(opt.m):
                        opt.K[:, a] = SVC.kernelTrans(opt.X, opt.X[a, :], self.kernel, self.gamma, self.coef0,
                                                      self.degree)
                    counter = Counter(opt.Y.flatten())
                    result_dict = self._solve_binary(counter, opt)
                    if self.alpha_ is None:
                        self.alpha_ = result_dict['alpha']
                        self.support_vectors_ = result_dict['support_vectors']
                        if self.probability:
                            self.K_ = result_dict['K']
                    else:
                        self.alpha_ = np.vstack((self.alpha_, result_dict['alpha']))
                        self.support_vectors_ = np.vstack((self.support_vectors_, result_dict['support_vectors']))
                        if self.probability:
                            self.K_ = np.vstack((self.K_, result_dict['K']))
                    self.b_[k] = result_dict['b']
                    self.w_[k] = result_dict['w']
                    self.alpha_num_[k] = len(result_dict['alpha'])
                    self.n_support_[k] = len(result_dict['support_vectors'])
                    if self.probability:
                        self.prob_A_[k] = result_dict['prob_A']
                        self.prob_B_[k] = result_dict['prob_B']
                    k += 1

        self.alpha_ = np.array(self.alpha_)
        self.support_vectors_ = np.array(self.support_vectors_)
        if self.probability:
            self.K_ = np.array(self.K_)

    def _weight(self, X, Y, alpha):
        # w = sum(alpha_i*yi*Xi)
        w = (Y * alpha * X).sum(axis=0)
        return w

    def _calc_prob(self, score, Y):
        """
        :param score:决策函数输出 sum(alpha*Y*K(X,X))+b
        :param Y:样本标签 {1, -1}
        :return A, B:sigmoid函数的参数 A, B
        """
        t = np.zeros(Y.shape)

        maxIter = 100
        minStep = 1e-10
        sigma = 1e-12

        numPositive = np.count_nonzero(Y == 1)
        numNegative = np.count_nonzero(Y == -1)
        length = numPositive + numNegative

        highTarget = (numPositive + 1.0) / (numPositive + 2.0)
        lowTarget = 1 / (numNegative + 2.0)
        for i in range(length):
            if Y[i] > 0:
                t[i] = highTarget
            else:
                t[i] = lowTarget

        A = 0.0
        B = np.log((numNegative + 1.0) / (numPositive + 1.0))
        f_val = 0.0
        for i in range(length):
            fApB = A * score[i] + B
            if fApB >= 0:
                f_val += t[i] * fApB + np.log(1 + np.exp(-fApB))
            else:
                f_val += (t[i] - 1) * fApB + np.log(1 + np.exp(fApB))

        it = 0
        while it < maxIter:
            if self.verbose:
                print("Probability: iter:{}".format(it))
            h11 = sigma
            h22 = sigma
            h21 = 0.0
            g1 = 0.0
            g2 = 0.0
            for i in range(length):
                fApB = A * score[i] + B
                if fApB >= 0:
                    p = np.exp(-fApB) / (1.0 + np.exp(-fApB))
                    q = 1.0 / (1.0 + np.exp(-fApB))
                else:
                    p = 1.0 / (1.0 + np.exp(fApB))
                    q = np.exp(fApB) / (1.0 + np.exp(fApB))
                d2 = p * q
                h11 += score[i] * score[i] * d2
                h22 += d2
                h21 += score[i] * d2
                d1 = t[i] - p
                g1 += score[i] * d1
                g2 += d1
            if np.abs(g1) < 1e-5 and np.abs(g2):
                break
            det = h11 * h22 - h21 * h21
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB
            stepSize = 1
            while stepSize >= minStep:
                newA = A + stepSize * dA
                newB = B + stepSize * dB
                new_f = 0.0
                for i in range(length):
                    fApB = score[i] * newA + newB
                    if fApB >= 0:
                        new_f += t[i] * fApB + np.log(1 + np.exp(-fApB))
                    else:
                        new_f = (t[i] - 1) * fApB + np.log(1 + np.exp(fApB))
                if new_f < f_val + 0.0001 * stepSize * gd:
                    A = newA
                    B = newB
                    f_val = new_f
                    break
                else:
                    stepSize /= 2.0
            if stepSize < minStep:
                print("Probability: Line search fails")
                break
            it += 1
        if it >= maxIter:
            print("Probability: Reaching maximum iterations")
        return A, B

    def predict(self, X_test):
        assert self.w_ is not None, "must fit before predict"
        # assert X_test.shape[1] == self.w_.shape[1], "the feature number of X_predict must be equal to X_train"
        num_classes = len(self.classes_)
        # 二分类预测
        if num_classes == 2:
            Y_predict, probability = self._predict_binary(self._X, X_test, self.alpha_, self._Y, self.b_, self.prob_A_,
                                                          self.prob_B_)
        # 多分类预测
        else:
            if self.decision_function_shape == 'ovr':
                Y_predict, probability = self._predict_ovr(X_test)
            else:
                Y_predict, probability = self._predict_ovo(X_test)
        if self.probability:
            self.predict_prob_ = probability
        return Y_predict

    def _predict_binary(self, X_train, X_test, alpha, Y, b, prob_A_, prob_B_):
        # 二分类预测
        # Y_predict = sign(sum(alpha*Y*K(X,X_test))+b)
        K = np.zeros((X_train.shape[0], X_test.shape[0]))
        for i in range(X_test.shape[0]):
            K[:, i] = self.kernelTrans(X_train, X_test[i, :], self.kernel, self.gamma, self.coef0, self.degree)
        score = (alpha * Y * K).sum(axis=0) + b
        Y_predict = (score >= 0).astype(int) * 2 - 1
        # 计算输出为正例1的概率
        predict_prob = None
        if self.probability:
            predict_prob = 1 / (1 + np.exp(prob_A_ * score + prob_B_))
        return Y_predict, predict_prob

    def _predict_ovr(self, X_test):
        # ovr多分类预测
        num_classes = len(self.classes_)
        # Y_predict_list = np.zeros(num_classes)
        if self.probability:
            predict_prob = np.zeros((X_test.shape[0], num_classes))

        k = 0
        for i in range(num_classes):
            # 整理对应的二分类数据
            alpha_num_before = np.sum(self.alpha_num_[:k])
            alpha_num = self.alpha_num_[k]
            alpha = self.alpha_[int(alpha_num_before):int(alpha_num_before + alpha_num)]
            X = self._X.copy()
            Y = self._Y.copy()
            i_index = Y == self.classes_[i]
            not_i_index = Y != self.classes_[i]
            Y[i_index] = 1
            Y[not_i_index] = -1
            b = self.b_[k]
            prob_A = None
            prob_B = None
            if self.probability:
                prob_A = self.prob_A_[k]
                prob_B = self.prob_B_[k]
            # 二分类预测
            Y_predict_i, predict_prob_i = self._predict_binary(X, X_test, alpha, Y, b, prob_A, prob_B)
            # Y_predict_list[k] = Y_predict_i
            if self.probability:
                predict_prob[:, k] = predict_prob_i
            k += 1

        if self.probability:
            self.predict_prob_ = predict_prob
            # 取每行概率值最大的下标对应的类
            predict_index = np.argmax(self.predict_prob_, axis=1)
            Y_predict = np.array([[self.classes_[i[0]]] for i in predict_index])
        return Y_predict

    def _predict_ovo(self, X_test):
        # ovo多分类预测
        num_classes = len(self.classes_)
        Y_predict_list = np.zeros((X_test.shape[0], int(num_classes * (num_classes - 1) / 2)))
        if self.probability:
            predict_prob = np.zeros((X_test.shape[0], int(num_classes * (num_classes - 1) / 2)))

        k = 0
        for i in range(num_classes):
            for j in range(num_classes):
                # 只预测上三角
                if i < j:
                    # 整理对应的二分类数据
                    alpha_num_before = np.sum(self.alpha_num_[:k])
                    alpha_num = self.alpha_num_[k]
                    alpha = self.alpha_[int(alpha_num_before):int(alpha_num_before + alpha_num)]
                    mask = self._Y != self.classes_[i]
                    mask[(self._Y != self.classes_[j]) == False] = False
                    X = np.delete(self._X, mask.flatten(), axis=0)
                    Y = np.delete(self._Y, mask.flatten(), axis=0)
                    i_index = Y == self.classes_[i]
                    j_index = Y == self.classes_[j]
                    Y[i_index] = 1
                    Y[j_index] = -1
                    b = self.b_[k]
                    prob_A = None
                    prob_B = None
                    if self.probability:
                        prob_A = self.prob_A_[k]
                        prob_B = self.prob_B_[k]
                    # 二分类预测
                    Y_predict_ij, predict_prob_ij = self._predict_binary(X, X_test, alpha, Y, b, prob_A, prob_B)
                    Y_predict_list[:, k] = Y_predict_ij
                    if self.probability:
                        predict_prob[:, k] = predict_prob_ij
                k += 1

        if self.probability:
            self.predict_prob_ = predict_prob
        Y_predict = (np.sum(Y_predict_list, axis=1) >= 0).astype(int) * 2 - 1
        return Y_predict

    def score(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        return accuracy_score(Y_test, Y_predict)

    def __repr__(self):
        return "SVC(C={}, gamma={})".format(self.C, self.gamma)
