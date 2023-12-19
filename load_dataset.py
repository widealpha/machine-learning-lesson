import matplotlib.pyplot as plt
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


if __name__ == '__main__':
    main()
