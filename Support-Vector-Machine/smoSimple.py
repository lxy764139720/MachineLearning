import numpy as np

from numpy import mat


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()

    b = 0
    m, n = np.shape(dataMatrix)
    alphas = mat(np.zeros((m, 1)))

    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # g(xi)=sum(alpha_i*yi*K(xi,x))+b
            g_xi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # Ei=g(xi)-yi
            Ei = g_xi - float(labelMat[i])

            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                g_xj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = g_xj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 计算上下界
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue

                # 更新alpha_j
                eta = dataMatrix[i, :] * dataMatrix[i, :].T + dataMatrix[j, :] * dataMatrix[j, :]. \
                    T - 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T
                if eta <= 0:
                    print("eta<=0")
                    continue
                alphas[j] += labelMat[j] * (Ei - Ej) / eta
                alphas[j] = min(alphas[j], H)
                alphas[j] = max(alphas[j], L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :]. \
                    T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :]. \
                    T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d".format(iter, i, alphaPairsChanged))

        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d".format(iter))
    return b, alphas
