import torchvision

# imageNetDataset = torchvision.datasets.ImageNet("datasetImageNet", "train", download=True,
#                                                 transform=torchvision.transforms.ToTensor)
from torch import nn

vgg_false = torchvision.models.vgg16(False)
vgg_true = torchvision.models.vgg16(True)
# vgg_false.classifier.add_module("add_new", nn.Linear(1000, 10))
vgg_false.classifier[6] = nn.Linear(4096, 10)
print(vgg_false)