import torchvision
from PIL import Image
from tensorboardX import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,transform=dataset_transform,download=True)

print(test_set[0])
writer = SummaryWriter("logs")
for i in range(100):
    img_tensor, target = test_set[i]
    writer.add_image("dataset",img_tensor,i)
writer.close()

