import torch
import torch.nn as nn

class Frequency_Attention(nn.Module):
    def __init__(self,channel,reduction=2):
        super(Frequency_Attention,self).__init__()
        self.avg_pool = nn.AvgPool2d(3,32)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Softmax(dim = 1)
            # nn.Sigmoid()
        )

    def forward(self,x):
        x1 = x.permute(0,3,1,2)
        y = self.avg_pool(x1).view(800,4)
        y = self.fc(y).view(800,1,1,4)
        return x * y.expand_as(x), y
"""
layer = Frequency_Attention(4)
a = torch.randn(800,3,32,4)
output,map = layer(a)
print(output.shape)
print(map[0])
"""