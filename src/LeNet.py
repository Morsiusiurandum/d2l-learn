import torch
from torch import nn

from helper import algorithm
from helper import utility
from helper import fashion_mnist


def train(net_input, train_iter_input, test_iter_input, num_epochs_input, lr_input, device_input):
    def init_weights(m):
        if type(m) is nn.Linear or type(m) is nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net_input.apply(init_weights)
    print('training on', device_input)
    net_input.to(device_input)
    optimizer = torch.optim.SGD(net_input.parameters(), lr=lr_input)
    loss = nn.CrossEntropyLoss()
    animator = utility.Animator(xlabel='epoch', xlim=[1, num_epochs_input],
                                legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = utility.Timer(), len(train_iter)
    for epoch in range(num_epochs_input):
        # 训练损失之和，训练准确率之和，样本数
        metric = utility.Accumulator(3)
        net_input.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device_input), y.to(device_input)
            y_hat = net_input(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], algorithm.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = algorithm.evaluate_accuracy_gpu(net_input, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs_input / timer.sum():.1f} examples/sec 'f'on {str(device_input)}')


net = nn.Sequential(

    # 5x5卷积,周围填充两个像素,输出6张10x10的特征图
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),

    # 2x2平均池化,步长为2,每张图像缩小为14x14
    nn.AvgPool2d(kernel_size=2, stride=2),

    # 5x5卷积,每输入6张输出16张10x10的特征图
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),

    # 2x2平均池化,每张图像缩小为5x5
    nn.AvgPool2d(kernel_size=2, stride=2),

    # 全部展开为一维数据,共5x5x16=400个输入,准备全连接
    nn.Flatten(),

    # 第一层400->120
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),

    # 第二层120->84
    nn.Linear(120, 84),
    nn.Sigmoid(),

    # 第三层84->10
    nn.Linear(84, 10),
)

batch_size: int = 256
train_iter, test_iter = fashion_mnist.load_data(batch_size_input=batch_size)
lr: float = 0.9
num_epochs: int = 10

train(net, train_iter, test_iter, num_epochs, lr, utility.try_gpu())
