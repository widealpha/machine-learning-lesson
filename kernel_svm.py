import pickle

from sklearn import metrics
from sklearn import svm

from load_dataset import split_train_test_dataset

if __name__ == '__main__':
    # 提取训练集
    x_train, y_train, x_test, y_test = split_train_test_dataset()
    svc = svm.SVC(kernel='rbf')
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
