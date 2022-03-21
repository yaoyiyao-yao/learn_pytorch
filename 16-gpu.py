import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torch import nn
import time

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        x = self.Model(x)
        return x

train_dataset = torchvision.datasets.CIFAR10("dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练数据长度：{}".format(train_data_size))
print("测试数据长度：{}".format(test_data_size))

train_loader = DataLoader(dataset=train_dataset, batch_size=64)
test_loader = DataLoader(dataset=test_dataset, batch_size=64)

net = Net()
if torch.cuda.is_available():
    net = net.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
learning_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), learning_rate)

epoch = 10
total_train_step = 0

writer = SummaryWriter("logs")

start_time = time.time()
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = net(imgs)
        optimizer.zero_grad()
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("time:{}".format(end_time - start_time))
            print("训练次数：{}，loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss, total_train_step)

    total_test_loss = 0.0
    total_correct_nums = 0.0
    with torch.no_grad():
        for test_data in test_loader:
            imgs, targets = test_data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = net(imgs)
            preds = output.argmax(1)
            total_correct_nums += (preds == targets).sum()
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
        writer.add_scalar("test_loss", total_test_loss, i)
        print("test_loss:{}".format(total_test_loss))
        print("test_accuracy", total_correct_nums/test_data_size)

    torch.save(net, "net{}".format(i))





