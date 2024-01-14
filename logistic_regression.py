import pickle
import time

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from load_dataset import split_train_test_dataset

if __name__ == '__main__':
    # 提取训练集
    x_train, y_train, x_test, y_test = split_train_test_dataset()

    start_time = time.time()
    lr = LogisticRegression(C=0.1, max_iter=100)
    lr.fit(x_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time

    # 使用测试集进行预测
    y_pred = lr.predict(x_test)

    # 计算准确率
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'训练时间${execution_time}秒\n准确率: {accuracy}')

    with open('model/logistic_regression.model', 'wb') as f:
        pickle.dump(lr, f)
