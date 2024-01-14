import torch
import torch.nn as nn
from torch import optim

from load_dataset import get_dataloader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入大小 (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 灰度图
                out_channels=16,  # 要得到几多少个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  #
            ),  # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten操作，结果为：(batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


if __name__ == '__main__':
    # 定义超参数
    num_epochs = 3  # 训练的总循环周期
    batch_size = 64  # 一个撮（批次）的大小，64张图片
    train_loader = get_dataloader(train=True, batch_size=batch_size)
    test_loader = get_dataloader(train=False, batch_size=batch_size)
    # 实例化
    net = CNN()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器，普通的随机梯度下降算法

    # 开始训练循环
    for epoch in range(num_epochs):
        # 当前epoch的结果保存下来
        train_rights = []

        for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
            net.train()
            output = net(data)

            loss = criterion(output, target)
            optimizer.zero_grad()  # 梯度归o
            loss.backward()
            optimizer.step()  # 更新优化器的学习率
            right = accuracy(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:

                net.eval()
                val_rights = []

                for (data, target) in test_loader:
                    output = net(data)
                    right = accuracy(output, target)
                    val_rights.append(right)

                # 准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print(
                    '当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                        epoch, batch_idx * batch_size, len(train_loader.dataset), 100.0 * batch_idx / len(train_loader),
                        loss.data, 100. * train_r[0] / train_r[1], 100. * val_r[0] / val_r[1]))
    torch.save(net, './model/gnn.model')