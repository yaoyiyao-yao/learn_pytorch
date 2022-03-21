import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        output = x +1
        return output

model = Model()
x = torch.tensor(1)
out = model(x)
print(out)