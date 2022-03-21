import torch
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=128)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x

network = Network()
optimizer = optim.SGD(network.parameters(), lr=0.01)
for opoch in range(20):
    running_loss = 0.0
    for loader in dataloader:
        imgs, targets = loader
        optimizer.zero_grad()
        output = network(imgs)
        loss_dataset = nn.CrossEntropyLoss()
        result_loss = loss_dataset(output, targets)
        running_loss += result_loss
        result_loss.backward()
        optimizer.step()
    print(running_loss)