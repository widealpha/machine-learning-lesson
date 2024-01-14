import scipy.io as scio
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        # 读取mat形式的数据
        all_data = scio.loadmat(root + '/mnist_all.mat')
        image_labels = []
        image_data = []
        # 读取训练集,测试集,以及对应的标签,存入image_data和image_labels中
        if train:
            pre = 'train'
        else:
            pre = 'test'
        for i in range(10):
            dict_key = pre + str(i)
            for data in all_data.get(dict_key):
                image_data.append(data.reshape(28, 28))
                image_labels.append(i)
        self.img_labels = image_labels
        self.img_data = image_data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
