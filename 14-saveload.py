import torchvision
import torch

vgg16 = torchvision.models.vgg16(pretrained=False)

# torch.save(vgg16, "vgg16_method1.pth")
#
# model = torch.load("vgg16_method1.pth")
# print(model)

# torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# model = torch.load("vgg16_method2.pth")
# print(model)

vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

