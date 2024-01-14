import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from mnist_dataset import MnistDataset


def main():
    dataloader = get_dataloader(train=True)
    # Display image and label.
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def load_dataset(train=True):
    dateset = MnistDataset(root='./dataset', train=train, transform=ToTensor())
    return dateset


def get_dataloader(train=True, batch_size=64, shuffle=True, num_workers=0):
    dataset = load_dataset(train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def split_train_test_dataset():
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
    main()
