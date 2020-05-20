from torch import nn

from model.resnet_3d import generate_model

class Resnet1013d(nn.Module):
    def __init__(self, num_classes=20, **kwargs):
        super().__init__()
        self.model = generate_model(101, n_classes=num_classes,**kwargs).cuda()

    def forward(self, x):
        # return self.model(x)
        # return torch.clamp(self.model(x), -1, 1)
        Tanh = nn.Tanh()
        return Tanh(self.model(x))
        # sigmoid = nn.Sigmoid()
        # return (sigmoid(self.model(x))-0.5)*2
