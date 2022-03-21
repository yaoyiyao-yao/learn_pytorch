import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

writer = SummaryWriter("logs")

step = 0
for epoch in range(2):
    step = 0
    for loader in test_loader:
        imgs ,targets = loader
        # imgs[0].shape: torch.Size([3, 32, 32])
        writer.add_images("test_epoch{}".format(epoch),imgs,step)
        step += 1
writer.close()
