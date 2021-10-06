import torch
import torch.nn as nn

class FrequencyAttentionLayer(nn.Module):
    def __init__(self,channel,reduction=2):
        super(FrequencyAttentionLayer,self).__init__()
        self.avg_pool = nn.AvgPool2d(3,32)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Softmax(dim = 1)
            # nn.Sigmoid()
        )

    def forward(self,x):
        batch = x.shape[0]
        x1 = x.permute(0,3,1,2)
        y = self.avg_pool(x1).view(batch,4)
        y = self.fc(y).view(batch,1,1,4)
        return x * y.expand_as(x), y
"""
layer = FrequencyAttentionLayer(4)
a = torch.randn(800,3,32,4)
output,map = layer(a)
print(output.shape)
print(map[0])
"""