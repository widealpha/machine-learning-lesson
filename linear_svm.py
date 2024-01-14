import pickle

import numpy as np
from scipy.io import loadmat
from sklearn import metrics
from sklearn import svm


# 将PyTorch数据加载器中的数据转换为NumPy数组
def gen_train_test_data():
    mnist_data = loadmat('./dataset/mnist_all.mat')
    labels = range(0, 10)
    x_train_arr = []
    y_train_arr = []
    x_test_arr = []
    y_test_arr = []
    for label in labels:
        train_data = mnist_data[f'train{label}']
        x_train_arr.extend(train_data)
        y_train_arr.extend([label] * len(train_data))

        train_data = mnist_data[f'test{label}']
        x_test_arr.extend(train_data)
        y_test_arr.extend([label] * len(train_data))

    return np.array(x_train_arr), np.array(y_train_arr), np.array(x_test_arr), np.array(y_test_arr),


if __name__ == '__main__':
    # 提取训练集
    x_train, y_train, x_test, y_test = gen_train_test_data()
    svc = svm.SVC(kernel='linear')
    svc.fit(x_train, y_train)

    # 使用测试集进行预测
    y_pred = svc.predict(x_test)

    # 计算准确率
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'准确率: {accuracy}')
    # # 可以进行其他性能评估，如混淆矩阵
    # confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print('混淆矩阵:')
    # print(confusion_matrix)
    with open('model/linear_svm.model', 'wb') as f:
        pickle.dump(svc, f)
