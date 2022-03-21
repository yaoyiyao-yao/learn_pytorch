import torch
from torch import nn

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
            # nn.Linear(1024, 64),
            # nn.Linear(64, 10)
        )
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.module1(x)
        print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

network = Network()
print(network)
input = torch.ones((64, 3, 32, 32))
output = network(input)
# print(output)
print(output.shape)