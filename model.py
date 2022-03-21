import torch
from torch import nn

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

if __name__ == '__main__':
    x = torch.ones(64, 3, 32, 32)
    net = Net()
    output = net(x)
    print(output.shape)