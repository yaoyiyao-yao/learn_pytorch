import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(196608, 16)

    def forward(self, x):
        x = self.linear(x)
        return x

network = Network()
for loader in dataloader:
    # imgs.shape: torch.Size([64, 3, 32, 32])
    imgs, target = loader
    # torch.Size([196608])
    imgs = torch.flatten(imgs)
    output = network(imgs)
    # torch.Size([16])
    print(output.shape)
