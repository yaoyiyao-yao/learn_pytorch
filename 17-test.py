import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        x = self.Model(x)
        return x

model = torch.load("net0")

img_path = "test_imgs/dog.png"
img = Image.open(img_path)
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
img_tensor = trans(img)
print(img_tensor.shape)
img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
print(img_tensor.shape)
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    print(output)
    print(output.argmax(1))






