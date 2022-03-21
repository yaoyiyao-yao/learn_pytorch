import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinear = nn.Sigmoid()
    def forward(self, x):
        x = self.nonlinear(x)
        return x

network = Network()
writer = SummaryWriter("logs")
step = 0
for loader in dataloader:
    imgs, targets = loader
    writer.add_images("ori", imgs, step)
    output = network(imgs)
    writer.add_images("new", output, step)
    step += 1
writer.close()
