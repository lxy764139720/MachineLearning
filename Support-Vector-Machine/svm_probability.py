import numpy as np


def probability(score, Y):
    """
    :param score:决策函数输出 sum(alpha*Y*K(X,X_test))+b
    :param Y:样本标签{1, -1}
    :return A, B:sigmoid函数的参数A, B
    """
    t = np.zeros(Y.shape)

    maxIter = 100
    minStep = 1e-10
    sigma = 1e-12

    numPositive = np.count_nonzero(Y == 1)
    numNegative = np.count_nonzero(Y != 1)
    length = numPositive + numNegative

    highTarget = (numPositive + 1.0) / (numPositive + 2.0)
    lowTarget = 1 / numNegative + 2.0
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
            print("Line search fails")
            break
        it += 1
    if it >= maxIter:
        print("Reaching maximum iterations")
    return A, B
