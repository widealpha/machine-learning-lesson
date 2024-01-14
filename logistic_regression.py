import pickle
import time

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from load_dataset import split_train_test_dataset

if __name__ == '__main__':
    # 提取训练集
    x_train, y_train, x_test, y_test = split_train_test_dataset()
    C = 0.1
    max_iter = 100
    print('Logistic Regression方法 使用 C=%s, max_iter=%s' % (C, max_iter))
    start_time = time.time()
    lr = LogisticRegression(C=C, max_iter=max_iter)
    lr.fit(x_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time

    # 使用测试集进行预测
    y_train_pred = lr.predict(x_train)
    y_pred = lr.predict(x_test)

    # 计算准确率
    train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    print('训练时间: %.2f秒' % execution_time)
    print(f'训练集准确率: {train_accuracy}')
    print(f'测试集准确率: {test_accuracy}')

    with open('model/logistic_regression.model', 'wb') as f:
        pickle.dump(lr, f)
