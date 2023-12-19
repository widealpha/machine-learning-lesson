import scipy.io as scio


def print_hi(name):
    data = scio.loadmat('./dataset/mnist_all.mat')
    print(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
