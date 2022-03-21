import torch
import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Nnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        output = self.maxpool(x)
        return output

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (1, 1, 5, 5))
# print(input.shape)
nnet = Nnet()
# output = nnet(input)
# print(output)
writer = SummaryWriter("logs")
step = 0
for loader in dataloader:
    imgs, target = loader
    writer.add_images("ori", imgs, step)
    # imgs.shape: torch.Size([64, 3, 32, 32])
    output = nnet(imgs)
    writer.add_images("show", output, step)
    step += 1
writer.close()
