import numpy as np
import matplotlib.pyplot as plt


def get_data(num_point, num_noise, X_small, Y_small, X_large, Y_large, seed):
    """
    生成环形数据与噪声
    @param num_point: 正常数据点的个数
    @param num_noise: 噪声的个数
    @param seed: 随机数种子。若seed值设置的一样，则每次随机生成的数据点都一样
    @return: 环形数据，噪声
    """
    np.random.seed(seed)
    point = []
    # 若生成的点的数量没有到达预期目标将不停迭代
    while len(point) != num_point:
        # 生成介于大矩形的均匀分布的数据点
        X_point = X_large[0] + (X_large[1] - X_large[0]) * np.random.rand()
        Y_point = Y_large[0] + (Y_large[1] - Y_large[0]) * np.random.rand()
        # 判断此点是否符合要求
        if (X_large[0] < X_point < X_small[0] or X_small[1] < X_point < X_large[1]
                or Y_large[0] < Y_point < Y_small[0] or Y_small[1] < Y_point < Y_large[1]):
            point.append((X_point, Y_point))
    point = np.array(point)
    # 生成噪声
    noise_Xcor = (0, 3)
    noise_Ycor = (0, 3)
    noise = []
    while len(noise) != num_noise:
        X_noise = noise_Xcor[0] + (noise_Xcor[1] - noise_Xcor[0]) * np.random.rand()
        Y_noise = noise_Ycor[0] + (noise_Ycor[1] - noise_Ycor[0]) * np.random.rand()
        if not (X_large[0] < X_noise < X_small[0] or X_small[1] < X_noise < X_large[1]
                or Y_large[0] < Y_noise < Y_small[0] or Y_small[1] < Y_noise < Y_large[1]):
            noise.append((X_noise, Y_noise))
    return np.array(point), np.array(noise)


def draw(data_point1, data_point2, data_point3):
    """
    绘制图像
    :param data_point1: 环形数据1
    :param data_point2: 环形数据2
    :param data_point3: 环形数据3
    :return: 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(data_point1[:, 0], data_point1[:, 1], c='r', marker='.')
    plt.scatter(data_point2[:, 0], data_point2[:, 1], c='b', marker='.')
    plt.scatter(data_point3[:, 0], data_point3[:, 1], c='g', marker='.')
    plt.show()


if __name__ == '__main__':
    # 设置生成环形的坐标数据，即大矩形跟小矩形坐标
    X_small_1 = (0.7, 2.3)
    Y_small_1 = (0.7, 2.3)
    X_large_1 = (0.5, 2.5)
    Y_large_1 = (0.5, 2.5)

    X_small_2 = (0.5, 2.5)
    Y_small_2 = (0.5, 2.5)
    X_large_2 = (0.3, 2.7)
    Y_large_2 = (0.3, 2.7)

    X_small_3 = (0.3, 2.7)
    Y_small_3 = (0.3, 2.7)
    X_large_3 = (0.1, 2.9)
    Y_large_3 = (0.1, 2.9)

    data1, _ = get_data(50, 0, X_small_1, Y_small_1, X_large_1, Y_large_1, 666)
    data2, _ = get_data(50, 0, X_small_2, Y_small_2, X_large_2, Y_large_2, 665)
    data3, _ = get_data(50, 0, X_small_3, Y_small_3, X_large_3, Y_large_3, 664)
    # draw(data1, data2, data3)
    label1 = np.zeros(len(data1))
    label2 = np.zeros(len(data2))
    label3 = np.zeros(len(data3))
    label1[:] = 1
    label2[:] = 2
    label3[:] = 3
    data1 = np.hstack((data1, label1.reshape((-1, 1))))
    data2 = np.hstack((data2, label2.reshape((-1, 1))))
    data3 = np.hstack((data3, label3.reshape((-1, 1))))
    data = np.vstack((data1, data2, data3))
    # data = np.vstack((data1, data2))
    np.savetxt('./round_scatter.csv', data, delimiter=',', fmt='%f')
