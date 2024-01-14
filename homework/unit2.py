import numpy as np
from matplotlib import pyplot as plt


def main():
    w1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
    w2 = np.array([[4, 4], [6, 4], [6, 6], [4, 6]], dtype=np.float64)
    # 计算均值
    mu1 = np.mean(w1, axis=0)
    mu2 = np.mean(w2, axis=0)
    # 计算协方差
    sigma1 = np.cov(w1, rowvar=False, bias=True)
    sigma2 = np.cov(w2, rowvar=False, bias=True)
    # 计算逆协方差
    inv_sigma1 = np.linalg.inv(sigma1)
    inv_sigma2 = np.linalg.inv(sigma2)
    # 计算参数
    if np.array_equiv(inv_sigma1, inv_sigma2):
        a = (mu1 - mu2).T @ inv_sigma1
        b = -0.5 * mu1.T @ inv_sigma1 @ mu1 + 0.5 * mu2.T @ inv_sigma2 @ mu2

        # 定义贝叶斯判别界面方程
        def f(x, y):
            return a[0] * x + a[1] * y + b

        x, y = np.meshgrid(np.linspace(-1, 7, 400), np.linspace(-1, 7, 400))
        plt.contour(x, y, f(x, y), levels=[0], colors='black')
    else:
        pw1 = np.log(0.5)
        pw2 = np.log(0.5)

        def d1(x, y):
            return -0.5 * np.array([x - mu1[0], y - mu1[1]]).T @ inv_sigma1 @ np.array([x - mu1[0], y - mu1[1]]) + pw1

        def d2(x, y):
            return -0.5 * np.array([x - mu2[0], y - mu2[1]]).T @ inv_sigma2 @ np.array([x - mu2[0], y - mu2[1]]) + pw2

        def f(x, y):
            return d1(x, y) - d2(x, y)

        x, y = np.meshgrid(np.linspace(-1, 7, 400), np.linspace(-1, 7, 400))
        plt.contour(x, y, f(x, y), levels=[0], colors='black')

    # 绘制样本点
    plt.scatter(w1[:, 0], w1[:, 1], c='red', marker='o', label=r'w1')
    plt.scatter(w2[:, 0], w2[:, 1], c='blue', marker='x', label=r'w2')

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    # 设置图例和标签
    plt.title('贝叶斯判别界面')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == '__main__':
    main()
