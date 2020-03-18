import numpy as np


class SimpleLinearRegression:
    def __init__(self):
        """初始化一维线性回归模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练一维线性回归模型"""
        assert x_train.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train),\
            "the size of x_train must be equal to the size of y_train."

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        self.a_ = (x_train - x_mean).dot(y_train-y_mean) / \
            (x_train-x_mean).dot(x_train-x_mean)
        self.b_ = y_mean - self.a_*x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1,\
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x_single，返回x_single的预测结果值"""
        return self.a_*x_single+self.b_

    def score(self, x_test, y_test):
        """根据测试数据集 x_test 和 y_test 确定当前模型的准确度"""
        assert len(x_test) == len(y_test),\
            "the size of y_true must be equal to the size of y_predict"
        y_predict = self.predict(x_test)
        return 1 - self._mean_squared_error(y_test, y_predict)/np.var(y_test)

    def _mean_squared_error(self, y_true, y_predict):
        """计算y_true和y_predict之间的MSE"""
        return np.sum((y_true-y_predict)**2)/len(y_true)

    def __repr__(self):
        return "SimpleLinearRegression()"
