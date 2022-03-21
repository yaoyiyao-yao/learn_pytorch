import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x
network = Network()
writer = SummaryWriter("logs")
step = 0
for loader in dataloader:
    #img.shape:torch.Size([64, 3, 32, 32])
    img, target = loader
    #output.shape:torch.Size([64, 6, 30, 30])
    output = network(img)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", img, step)
    writer.add_images("output", output, step)
    step += 1
writer.close()
