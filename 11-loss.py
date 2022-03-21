import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([1, 2, 5], dtype=torch.float32)

x = torch.reshape(x, (1, 1, 3))
y = torch.reshape(y, (1, 1, 3))

loss = nn.L1Loss()
output = loss(x, y)
print(output)

loss_mse = nn.MSELoss()
output_mse = loss_mse(x, y)
print(output_mse)

x = torch.tensor([0.1, 0.2, 0.3])
target = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
output_cross = loss_cross(x, target)
print(output_cross)

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

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
for loader in dataloader:
    imgs, targets = loader
    output = network(imgs)
    loss_dataset = nn.CrossEntropyLoss()
    result_loss = loss_dataset(output, targets)
    result_loss.backward()
    print("zaijian")
